"""Live observer of one representative simulator environment during PPO training.

Architecture: N headless training environments -> learner updates the policy ->
periodic detached CPU snapshot -> this observer, which drives its own single
``VectorHillEnv`` with its own private ``ActorCritic`` copy on a daemon thread.
Nothing produced here feeds back into learning: the observer owns its seed, its
NumPy generator, its model, its counters and its sinks; the learner only hands it
an immutable payload through ``snapshot`` (an O(1) reference swap under a lock)
and never waits on it. Torch is imported lazily inside ``TrainingObserver``.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from gradientclimb.telemetry import gpu_sample

from .sinks import FanoutSink, FfmpegSink, FrameSink, NullSink, SinkClosed, TkSink

if TYPE_CHECKING:
    from PIL import Image

try:
    import psutil
except ImportError:  # CPU telemetry then renders as n/a, exactly like the recorder.
    psutil = None

# Held-out seeds: evaluate() defaults (1000-1019), generalization_suite()
# defaults (2000-2019), run_training validation (10000-10019), the evaluate/watch
# CLI defaults (20000-20019) and evaluate_checkpoints (30000-30019). The observer
# must never replay any of these.
RESERVED_SEED_RANGES = (
    (1000, 1020),
    (2000, 2020),
    (10000, 10020),
    (20000, 20020),
    (30000, 30020),
)
DEFAULT_OBSERVER_SEED = 41000
OBSERVER_MODES = ("thread",)
OBSERVER_DISPLAYS = ("window", "none")
OBSERVER_POLICIES = ("current", "best")
BEST_SELECTOR = "recent_training_mean_distance"
# Provenance of the snapshot whose weights are loaded; every per-episode record
# and the SNAPSHOT overlay line describe exactly these values.
PROVENANCE_KEYS = (
    "index",
    "elapsed",
    "optimizer_updates",
    "environment_steps",
    "episodes",
    "iterations",
    "mean_episode_distance",
)
LOG = logging.getLogger(__name__)


def seed_is_reserved(seed: int) -> bool:
    return any(low <= seed < high for low, high in RESERVED_SEED_RANGES)


@dataclass(frozen=True)
class ObserverConfig:
    """Observer options; never merged into the training hyperparameters."""

    mode: str = "thread"
    display: str = "none"
    policy: str = "current"
    snapshot_interval: float = 5.0
    seed: int = DEFAULT_OBSERVER_SEED
    fps: float | None = None
    video: str | None = None
    width: int = 960
    height: int = 540
    telemetry_interval: float = 2.0
    gpu_telemetry_interval: float | None = 10.0

    @classmethod
    def from_mapping(cls, value: ObserverConfig | dict | None) -> ObserverConfig | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("Observer options must be a mapping or ObserverConfig")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown observer options: {sorted(unknown)}")
        return cls(**value)

    def validate(self, training_seed: int) -> None:
        if self.mode not in OBSERVER_MODES:
            raise ValueError(f"Observer mode must be one of {OBSERVER_MODES}; got {self.mode!r}")
        if self.display not in OBSERVER_DISPLAYS:
            raise ValueError(f"Observer display must be one of {OBSERVER_DISPLAYS}")
        if self.policy not in OBSERVER_POLICIES:
            raise ValueError(f"Observer policy must be one of {OBSERVER_POLICIES}")
        if not isinstance(self.snapshot_interval, (int, float)) or not self.snapshot_interval > 0:
            raise ValueError("Observer snapshot_interval must be positive seconds")
        if self.fps is not None and (not isinstance(self.fps, (int, float)) or not self.fps > 0):
            raise ValueError("Observer fps must be a positive number or null")
        if not isinstance(self.telemetry_interval, (int, float)) or not self.telemetry_interval > 0:
            raise ValueError("Observer telemetry_interval must be positive seconds")
        gpu = self.gpu_telemetry_interval
        if gpu is not None and (not isinstance(gpu, (int, float)) or not gpu > 0):
            raise ValueError("Observer gpu_telemetry_interval must be positive seconds or null")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Observer seed must be a non-negative integer")
        if seed_is_reserved(self.seed):
            raise ValueError(
                f"Observer seed {self.seed} lies in a held-out evaluation range "
                f"{RESERVED_SEED_RANGES}; choose another seed"
            )
        if self.seed == training_seed:
            raise ValueError("Observer seed must differ from the training seed")
        if self.width < 16 or self.height < 16:
            raise ValueError("Observer frame size must be at least 16x16")
        if self.video is not None and not str(self.video):
            raise ValueError("Observer video path must be a non-empty string or null")

    def resolved_fps(self, action_duration: float) -> float:
        return float(self.fps) if self.fps else 1.0 / action_duration

    def to_record(self, action_duration: float) -> dict[str, Any]:
        return {
            "enabled": True,
            **asdict(self),
            "fps": self.resolved_fps(action_duration),
            "best_selector": BEST_SELECTOR,
            "learning_data": "excluded",
        }


def build_sinks(config: ObserverConfig) -> tuple[list[FrameSink], str | None]:
    """Sinks for the configured display; a missing FFmpeg skips the video gracefully.

    Called before any run directory exists: a window that cannot be opened
    (``TkSink.probe``) and an existing video file both raise here, so a run is
    never sealed after training for its whole budget without the requested view.
    """
    sinks: list[FrameSink]
    if config.display == "window":
        TkSink.probe()
        sinks = [TkSink("GradientClimb | live training observer")]
    else:
        sinks = [NullSink()]
    skipped = None
    if config.video:
        if shutil.which("ffmpeg") is None:
            skipped = "ffmpeg executable not found"
        else:
            # FileExistsError propagates: refuse to overwrite research evidence before training.
            sinks.append(FfmpegSink(config.video))
    return sinks, skipped


def cpu_percent_between(previous: Any, current: Any) -> float | None:
    """System CPU busy percentage between two ``psutil.cpu_times()`` snapshots.

    A private computation over two immutable snapshots: unlike
    ``psutil.cpu_percent(interval=None)`` it shares no process-global "last
    call" state, so it cannot perturb the run recorder's telemetry stream.
    """

    def busy_total(times: Any) -> tuple[float, float]:
        total = float(sum(times))
        total -= float(getattr(times, "guest", 0.0)) + float(getattr(times, "guest_nice", 0.0))
        busy = total - float(times.idle) - float(getattr(times, "iowait", 0.0))
        return busy, total

    busy_before, total_before = busy_total(previous)
    busy_after, total_after = busy_total(current)
    elapsed = total_after - total_before
    if elapsed <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (busy_after - busy_before) / elapsed))


class ResourceSampler(threading.Thread):
    """Sample CPU/GPU utilization on its own daemon thread for the overlay.

    CPU comes from private ``psutil.cpu_times()`` deltas (no shared psutil
    state); GPU/VRAM come from one ``nvidia-smi`` subprocess every
    ``gpu_interval`` seconds (``None`` disables the subprocess entirely).
    """

    def __init__(self, interval: float, gpu_interval: float | None = 10.0) -> None:
        super().__init__(name="gradientclimb-observer-telemetry", daemon=True)
        self.interval = float(interval)
        self.gpu_interval = None if gpu_interval is None else float(gpu_interval)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {
            "cpu_percent": None,
            "gpu_percent": None,
            "vram_used_bytes": None,
        }
        self.samples = 0
        self.gpu_samples = 0

    @property
    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def _cpu_times(self) -> Any:
        if psutil is None:
            return None
        try:
            return psutil.cpu_times()
        except (OSError, ValueError):
            return None

    def run(self) -> None:
        previous = self._cpu_times()
        next_gpu = 0.0
        elapsed = 0.0
        while True:
            update: dict[str, Any] = {}
            current = self._cpu_times()
            if previous is not None and current is not None:
                percent = cpu_percent_between(previous, current)
                if percent is not None:
                    update["cpu_percent"] = percent
            if current is not None:
                previous = current
            if self.gpu_interval is not None and elapsed >= next_gpu:
                try:
                    gpu = gpu_sample()
                except (OSError, ValueError):
                    gpu = {}
                update["gpu_percent"] = gpu.get("gpu_percent")
                update["vram_used_bytes"] = gpu.get("vram_used_bytes")
                self.gpu_samples += 1
                next_gpu = elapsed + self.gpu_interval
            with self._lock:
                self._latest.update(update)
                self.samples += 1
            if self._stop.wait(self.interval):
                return
            elapsed += self.interval

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout)


def _integer(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _number(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}{unit}"


def _bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / 1024**3:.1f} GB"


def format_overlay_lines(fields: dict[str, Any]) -> list[str]:
    """Four overlay lines; every missing or null field renders as n/a."""
    get = fields.get
    reward = get("step_reward")
    reward_text = "n/a" if reward is None else f"{float(reward):+.3f}"
    training = (
        "TRAINING  "
        f"t {_number(get('training_elapsed_seconds'))} s  "
        f"steps {_integer(get('environment_steps'))}  "
        f"episodes {_integer(get('episodes'))}  "
        f"updates {_integer(get('optimizer_updates'))}  "
        f"{_number(get('environment_steps_per_second'), 0)} steps/s  "
        f"ppo {get('algorithm_version') or 'n/a'} / {get('simulator_version') or 'n/a'}"
    )
    snapshot = (
        "SNAPSHOT  "
        f"#{_integer(get('snapshot_index'))} {get('snapshot_policy') or 'n/a'}  "
        f"taken {_number(get('snapshot_elapsed'))} s / "
        f"{_integer(get('snapshot_optimizer_updates'))} updates  "
        f"age {_number(get('snapshot_age_seconds'))} s  "
        f"best recent mean {_number(get('best_mean_episode_distance'), 1, ' m')}"
    )
    ended = get("observer_termination")
    observer = (
        "OBSERVER  "
        f"ep {_integer(get('observer_episode'))}  "
        f"distance {_number(get('observer_distance'), 1, ' m')}  "
        f"vx {_number(get('velocity'), 2, ' m/s')}  "
        f"theta {_number(get('theta'), 2, ' rad')}  "
        f"gas {_integer(get('gas'))}  brake {_integer(get('brake'))}  "
        f"reward {reward_text}  "
        f"step {_integer(get('observer_step'))}  frames {_integer(get('observer_frames'))}"
        + (f"  ended: {ended}" if ended else "")
    )
    system = (
        "SYSTEM    "
        f"cpu {_number(get('cpu_percent'), 0, '%')}  "
        f"gpu {_number(get('gpu_percent'), 0, '%')}  "
        f"vram {_bytes(get('vram_used_bytes'))}"
    )
    return [training, snapshot, observer, system]


_FONT = None


def _font():
    """The legacy bitmap font: ~1.4 ms per overlay against ~8 ms for FreeType text."""
    global _FONT
    if _FONT is None:
        from PIL import ImageFont

        loader = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)
        _FONT = loader()
    return _FONT


def compose_overlay(image: Image.Image, fields: dict[str, Any]) -> Image.Image:
    """Return a copy of ``image`` with a telemetry panel; the size never changes."""
    from PIL import ImageDraw

    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    lines = format_overlay_lines(fields)
    top, line_height, margin = 66, 17, 12
    panel_height = line_height * len(lines) + 12
    draw.rectangle(
        (margin, top, frame.width - margin, top + panel_height), fill=(16, 25, 39), outline=None
    )
    for index, line in enumerate(lines):
        draw.text((22, top + 6 + line_height * index), line, fill="#e2eaf7", font=_font())
    return frame


def loaded_provenance(payload: dict[str, Any], policy: str) -> tuple[dict[str, Any], dict]:
    """The weights a payload selects for ``policy`` and their own provenance.

    In ``best`` mode the provenance is that of the best snapshot
    (``best_snapshot``), not of the payload that carried it: per-episode
    records and the overlay must describe the weights actually loaded.
    """
    if policy == "best" and payload.get("best_state_dict") is not None:
        provenance = dict(payload.get("best_snapshot") or {})
        provenance.setdefault("index", payload.get("best_index"))
        return provenance, payload["best_state_dict"]
    return {key: payload.get(key) for key in PROVENANCE_KEYS}, payload["state_dict"]


class TrainingObserver:
    """Drive one private environment with the newest training snapshot on a thread.

    Isolation guarantees: the environment, its NumPy generator, the model copy
    and every counter are private; construction forks the Torch RNG so building
    the private ``ActorCritic`` leaves the learner's global generator untouched;
    ``snapshot`` only swaps a reference under a lock; no rollout, reward or
    RNG state ever flows back to the learner. Weights are reloaded at observer
    episode boundaries (or immediately for the first snapshot). With
    ``policy="best"`` the observer follows ``best_state_dict`` — the snapshot
    with the highest recent *training* mean distance, not held-out evaluation.
    """

    def __init__(
        self,
        config: ObserverConfig,
        training: dict[str, Any],
        sinks: list[FrameSink],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import torch

        from gradientclimb.algorithms.ppo import ActorCritic
        from gradientclimb.simulation import SIMULATOR_VERSION, VectorHillEnv

        self.config = config
        self.training = dict(training)
        self.simulator_version = SIMULATOR_VERSION
        self.env = VectorHillEnv(
            1,
            config.seed,
            self.training.get("profile", "default"),
            self.training.get("terrain", "train"),
            bool(self.training.get("randomization", False)),
            int(self.training.get("stack", 4)),
            int(self.training.get("max_steps", 1000)),
        )
        # Orthogonal initialization would otherwise consume the learner's global RNG.
        with torch.random.fork_rng(devices=[]):
            self.model = ActorCritic(
                self.env.observation_dim, int(self.training.get("hidden_size", 64)), "cpu"
            )
        self.model.eval()
        self.fps = config.resolved_fps(self.env.action_duration)
        self.width, self.height = int(config.width), int(config.height)
        self._sink = FanoutSink(sinks)
        self._clock = clock
        # The default wait is interruptible by stop(); tests inject a fake clock.
        self._sleep = sleep if sleep is not None else self._wait
        self._lock = threading.Lock()
        self._pending: tuple[int, dict[str, Any], float] | None = None
        self._sequence = 0
        self._first = threading.Event()
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._sampler = ResourceSampler(config.telemetry_interval, config.gpu_telemetry_interval)
        self._latest: dict[str, Any] | None = None
        self._latest_received_at: float | None = None
        self._latest_sequence = 0
        self._loaded_sequence = 0
        self._loaded: dict[str, Any] | None = None
        self.loaded_snapshot_index: int | None = None
        self.snapshots_received = 0
        self.snapshots_loaded = 0
        self.frames_rendered = 0
        self.observer_env_steps = 0
        self.episodes: list[dict[str, Any]] = []
        self.sink_errors: list[str] = []
        self.error: str | None = None
        self.thread_joined = False
        self.stop_timeout: float | None = None
        self.observer_wall_seconds = 0.0
        self.render_seconds_total = 0.0
        self.inference_seconds_total = 0.0
        self._sink_report: dict[str, Any] | None = None
        self._started_at: float | None = None

    def _wait(self, seconds: float) -> None:
        self._stop.wait(seconds)

    # ----- learner side -------------------------------------------------------
    def snapshot(self, payload: dict[str, Any]) -> None:
        """Learner callback: swap a reference under a lock; never raises, never blocks."""
        try:
            with self._lock:
                self._sequence += 1
                self._pending = (self._sequence, payload, self._clock())
                self.snapshots_received += 1
            self._first.set()
        except Exception:  # The learner must never see an observer failure.
            LOG.warning("Observer snapshot handoff failed", exc_info=True)

    # ----- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("The observer thread was already started")
        self._thread = threading.Thread(
            target=self._run, name="gradientclimb-observer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> dict[str, Any]:
        """Stop rendering and join; ``timeout`` excludes the sinks' bounded close.

        The join budget is ``timeout`` plus the longest sink ``close_timeout``
        (an FFmpeg flush), so a finished encoder is registered rather than lost.
        """
        self._stop.set()
        self.stop_timeout = float(timeout)
        if self._thread is not None:
            self._thread.join(timeout + self._sink.close_timeout)
            self.thread_joined = not self._thread.is_alive()
        else:
            self.thread_joined = True
        return self.report()

    def report(self) -> dict[str, Any]:
        sink_report = self._sink_report or {}
        video = None
        video_skipped_reason = None
        for item in sink_report.get("sinks", []):
            if item.get("kind") == "ffmpeg" and "error" not in item:
                video = item.get("video")
        if self.stop_timeout is not None and not self.thread_joined:
            budget = self.stop_timeout + self._sink.close_timeout
            video = None  # The encoder may still be flushing; nothing is registrable yet.
            video_skipped_reason = (
                f"observer thread still running {budget:g} s after stop; "
                "counters are a mid-flight reading and no video was registered"
            )
        return {
            "mode": self.config.mode,
            "display": self.config.display,
            "policy": self.config.policy,
            "seed": self.config.seed,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "snapshots_received": self.snapshots_received,
            "snapshots_loaded": self.snapshots_loaded,
            "loaded_snapshot_index": self.loaded_snapshot_index,
            "frames_rendered": self.frames_rendered,
            "observer_env_steps": self.observer_env_steps,
            "observer_episodes": len(self.episodes),
            "episodes": [dict(row) for row in self.episodes],
            "video": video,
            "video_skipped_reason": video_skipped_reason,
            "sinks": list(sink_report.get("sinks", [])),
            "sink_errors": list(self.sink_errors) + list(sink_report.get("errors", [])),
            "error": self.error,
            "stopped_cleanly": self.error is None
            and self.thread_joined
            and (self._thread is None or self._finished.is_set()),
            "thread_joined": self.thread_joined,
            "observer_wall_seconds": self.observer_wall_seconds,
            "render_seconds_total": self.render_seconds_total,
            "inference_seconds_total": self.inference_seconds_total,
            "telemetry_samples": self._sampler.samples,
            "gpu_telemetry_samples": self._sampler.gpu_samples,
            "best_selector": BEST_SELECTOR,
            "learning_data": "excluded",
        }

    # ----- observer thread ----------------------------------------------------
    def _peek(self) -> None:
        """Refresh the overlay's view of training progress; O(1) under the lock."""
        with self._lock:
            pending = self._pending
        if pending is not None and pending[0] > self._latest_sequence:
            self._latest_sequence = pending[0]
            self._latest, self._latest_received_at = pending[1], pending[2]

    def _load_if_newer(self) -> None:
        self._peek()
        if self._latest is not None and self._latest_sequence > self._loaded_sequence:
            self._load(self._latest, self._latest_sequence)

    def _load(self, payload: dict[str, Any], sequence: int) -> None:
        provenance, state = loaded_provenance(payload, self.config.policy)
        index = provenance.get("index")
        self._loaded_sequence = sequence
        self._loaded = provenance
        if index != self.loaded_snapshot_index or self.snapshots_loaded == 0:
            self.model.load_state_dict(state)
            self.loaded_snapshot_index = index
            self.snapshots_loaded += 1

    def _training_elapsed(self, now: float) -> float | None:
        latest = self._latest or {}
        elapsed, received = latest.get("elapsed"), self._latest_received_at
        if elapsed is None or received is None:
            return None
        return elapsed + (now - received)

    def _fields(self, reward: float, episode: dict[str, Any] | None = None) -> dict[str, Any]:
        latest = self._latest or {}
        loaded = self._loaded or {}
        now = self._clock()
        training_elapsed = self._training_elapsed(now)
        steps = latest.get("environment_steps")
        elapsed = latest.get("elapsed")
        rate = steps / elapsed if steps is not None and elapsed else None
        # Age on the training clock of the snapshot actually loaded, whichever
        # payload carried it; in current mode this equals the wall time since arrival.
        loaded_elapsed = loaded.get("elapsed")
        age = (
            training_elapsed - loaded_elapsed
            if training_elapsed is not None and loaded_elapsed is not None
            else None
        )
        resources = self._sampler.latest
        if episode is None:
            observer_state = {
                "observer_distance": float(self.env.best_x[0]),
                "velocity": float(self.env.vx[0]),
                "theta": float(self.env.theta[0]),
                "gas": int(self.env.pedals[0, 0]),
                "brake": int(self.env.pedals[0, 1]),
                "observer_step": int(self.env.steps[0]),
                "observer_termination": None,
            }
        else:
            # The simulator reset in the same step: describe the finished episode
            # from its record rather than the freshly reset car.
            observer_state = {
                "observer_distance": float(episode["distance"]),
                "velocity": None,
                "theta": None,
                "gas": episode.get("gas"),
                "brake": episode.get("brake"),
                "observer_step": int(episode["length"]),
                "observer_termination": episode["termination"],
            }
        return {
            "training_elapsed_seconds": training_elapsed,
            "environment_steps": steps,
            "episodes": latest.get("episodes"),
            "optimizer_updates": latest.get("optimizer_updates"),
            "environment_steps_per_second": rate,
            "snapshot_index": self.loaded_snapshot_index,
            "snapshot_policy": self.config.policy,
            "snapshot_elapsed": loaded_elapsed,
            "snapshot_optimizer_updates": loaded.get("optimizer_updates"),
            "snapshot_age_seconds": age,
            "best_mean_episode_distance": latest.get("best_mean_episode_distance"),
            "observer_episode": len(self.episodes),
            **observer_state,
            "step_reward": float(reward),
            "observer_frames": self.frames_rendered,
            "cpu_percent": resources.get("cpu_percent"),
            "gpu_percent": resources.get("gpu_percent"),
            "vram_used_bytes": resources.get("vram_used_bytes"),
            "simulator_version": self.simulator_version,
            "algorithm_version": latest.get("algorithm_version"),
        }

    def _episode_record(self, episode: dict[str, Any]) -> dict[str, Any]:
        loaded = self._loaded or {}
        return {
            "observer_episode": len(self.episodes),
            "snapshot_index": self.loaded_snapshot_index,
            "snapshot_elapsed": loaded.get("elapsed"),
            "snapshot_optimizer_updates": loaded.get("optimizer_updates"),
            "snapshot_environment_steps": loaded.get("environment_steps"),
            "policy": self.config.policy,
            "distance": float(episode["distance"]),
            "return_": float(episode["return_"]),
            "length": int(episode["length"]),
            "seconds": float(episode["seconds"]),
            "termination": episode["termination"],
            "terminated": bool(episode["terminated"]),
            "truncated": bool(episode["truncated"]),
            "observer_wall_seconds": self._clock() - self._started_at,
        }

    def _run(self) -> None:
        self._started_at = self._clock()
        try:
            self._sink.open(self.width, self.height, self.fps)
            self._sampler.start()
            observations, _ = self.env.reset(self.config.seed)
            while not self._stop.is_set() and not self._first.wait(0.05):
                pass
            if self._stop.is_set():
                return
            self._load_if_newer()
            interval = 1.0 / self.fps
            # Even when behind schedule the observer yields the GIL between frames.
            minimum_yield = min(0.005, interval * 0.1)
            next_frame = self._clock()
            while not self._stop.is_set():
                t = time.perf_counter()
                action = self.model.act(observations, deterministic=True)
                self.inference_seconds_total += time.perf_counter() - t
                joint = int(action[0])
                pedals = (joint & 1, (joint >> 1) & 1)  # gas bit, brake bit
                observations, reward, _terminated, _truncated, info = self.env.step(action)
                self.observer_env_steps += 1
                episode = dict(info["episodes"][0]) if info["episodes"] else None
                if episode is not None:
                    episode["gas"], episode["brake"] = pedals
                self._peek()
                fields = self._fields(float(reward[0]), episode)
                t = time.perf_counter()
                frame = compose_overlay(self.env.render(0, self.width, self.height), fields)
                self.render_seconds_total += time.perf_counter() - t
                try:
                    self._sink.write(frame)
                except SinkClosed as error:
                    self.sink_errors.append(str(error))
                    break
                self.frames_rendered += 1
                if episode is not None:
                    self.episodes.append(self._episode_record(episode))
                    self._load_if_newer()
                next_frame += interval
                delay = next_frame - self._clock()
                if delay <= 0:
                    next_frame = self._clock()  # fell behind: drop the deficit, never spin
                self._sleep(max(delay, minimum_yield))
        except Exception as error:  # Recorded, never raised into the learner.
            self.error = f"{type(error).__name__}: {error}"
            LOG.warning("Training observer stopped on an error", exc_info=True)
        finally:
            try:
                self._sink_report = self._sink.close()
            except Exception as error:  # A sink failure is evidence, not a learner error.
                self.sink_errors.append(f"close: {error}")
                LOG.warning("Observer sink close failed", exc_info=True)
            self._sampler.stop()
            self.observer_wall_seconds = self._clock() - self._started_at
            self._finished.set()
