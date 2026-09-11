"""Build and optionally execute two bounded, source-backed analysis notebooks.

No model inference or training is performed. Canonical run records remain unchanged.
Use --execute to save actual cell outputs and HTML previews under artifacts/notebooks.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import nbformat

PROJECT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", "notebook_cell.py", "-"],
        input=dedent(text).strip(),
        text=True,
        capture_output=True,
        check=True,
        cwd=PROJECT,
    ).stdout
    return nbformat.v4.new_code_cell(formatted.rstrip())


SETUP = """
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, display

PROJECT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'src/gradientclimb').is_dir())
ROOT = PROJECT / 'artifacts'
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))
# ANALYSIS_IMPORTS

PRIMARY = 'c0a9e142-1ad6-4d88-810d-bda4ca297f40'
CHECKPOINT_EVALUATION = 'b8df5a43-deb0-4b3a-917b-2398322c327c'
GENERALIZATION_EVALUATION = '3a9a0c20-e544-4efd-9c68-e2830d8e8224'
plt.rcParams.update({'figure.figsize': (9, 4.5), 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True})
COLORS = {'source': '#087f8c', 'shift': '#bc5a36', 'reference': '#596b75'}

def show_table(headers, rows):
    heading = ''.join('<th style="text-align:left;padding:7px">'+html.escape(str(x))+'</th>' for x in headers)
    body = ''.join('<tr>'+''.join('<td style="padding:7px;border-top:1px solid #ddd;overflow-wrap:anywhere">'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)
    display(HTML('<div style="overflow:auto"><table style="border-collapse:collapse;font-size:13px"><thead><tr>'+heading+'</tr></thead><tbody>'+body+'</tbody></table></div>'))

primary = load_run(ROOT, PRIMARY)
assert primary['status'] == 'completed'
assert verify_run(ROOT, PRIMARY)['valid']
display(Markdown(f"**Primary source:** `{PRIMARY}`. Actual training: **{primary['summary']['training_clock_seconds']:.3f} s**; final validation mean **{primary['summary']['mean_distance']:.2f} nominal m**. Real-game qualification remains incomplete."))
cycle = json.loads((PROJECT/'research/experiments/cycle-1-plan-status.json').read_text())
audit = json.loads((PROJECT/'research/experiments/cycle-1-integrity.json').read_text())
assert audit['run_count'] == audit['valid_count'] and not audit['unfinished_run_ids']
for checked in audit['runs']:
    assert hashlib.sha256((ROOT/'runs'/checked['run_id']/'seal.json').read_bytes()).hexdigest() == checked['seal_sha256']
display(Markdown(f"**Cycle 1 is paused.** Evidence cutoff: **{cycle['evidence_cutoff_utc']}**. The full integrity audit covers **{audit['valid_count']}** finalized records; current seal bytes match. The registered post-hour battery completed; the student pilot never started. This notebook performs record analysis only."))
"""


def checkpoint_notebook():
    return [
        markdown("""
        # Checkpoint quality and the learning clock
        ## Summary
        The sealed primary record contains a real one-hour simulator training clock. This notebook queries exact saved checkpoint times and separates fixed-policy evaluation from changing-policy training diagnostics. No new model evaluation occurs here.
        """),
        code(
            SETUP.replace(
                "# ANALYSIS_IMPORTS",
                "from gradientclimb.evaluation import compare_paired\nfrom gradientclimb.experiments import connect_database, load_run, verify_run",
            )
        ),
        markdown("""
        ## Context & methods
        All distances are nominal units in the uncalibrated surrogate. The six checkpoints share validation seeds 10000–10019; they are correlated policies from one training seed, not six independent training replicates. Bootstrap intervals condition on the fixed policies and sampled evaluation episodes.
        ### Key assumptions
        Canonical seals establish file integrity, not real-game fidelity. Imports, provenance and offline evaluation lie outside the recorded monotonic training clock. A final score never substitutes for an earlier time point.
        ## Data
        ### 1. Verify the exact source records
        """),
        code("""
        evaluation = load_run(ROOT, CHECKPOINT_EVALUATION)
        assert evaluation['parent_run'] == PRIMARY and verify_run(ROOT, CHECKPOINT_EVALUATION)['valid']
        source_ids = [PRIMARY, CHECKPOINT_EVALUATION]
        show_table(['Run', 'run.json SHA-256', 'Status'], [
            [run_id, hashlib.sha256((ROOT/'runs'/run_id/'run.json').read_bytes()).hexdigest(), load_run(ROOT, run_id)['status']]
            for run_id in source_ids
        ])
        """),
        markdown(
            "### 2. Query the recorded training curve\nThis parameterized DuckDB query uses the public canonical-store API. It reads a snapshot without locking the active reproduction writer."
        ),
        code("""
        sql = '''
        SELECT CAST(json_extract_string(dimensions_json, '$.training_elapsed_seconds') AS DOUBLE) AS training_seconds,
               value AS rolling_training_distance
        FROM metrics
        WHERE run_id = ? AND name = 'mean_episode_distance'
        ORDER BY training_seconds
        '''
        with connect_database(ROOT) as connection:
            connection.execute('SET threads = 1')
            training_curve = connection.execute(sql, [PRIMARY]).fetchall()
        assert training_curve and all(row[0] is not None for row in training_curve)
        print(f'{len(training_curve)} recorded diagnostic points; first/last actual seconds: {training_curve[0][0]:.3f}, {training_curve[-1][0]:.3f}')
        """),
        markdown("## Results\n### 3. Inspect exact checkpoint values"),
        code("""
        checkpoint_rows = sorted(evaluation['evaluation_results'], key=lambda row: row['metadata']['requested_minutes'])
        registered_hashes = {row['sha256'] for row in primary['artifact_manifest'] if row['kind'] == 'checkpoint'}
        for row in checkpoint_rows:
            assert row['checkpoint_hash'] in registered_hashes
            assert row['results']['seeds'] == list(range(10000, 10020))
            recorded = row['results']['summary']['distance']['mean']
            assert abs(sum(item['distance'] for item in row['results']['episodes']) / len(row['results']['episodes']) - recorded) < 1e-7
        show_table(['Requested min', 'Actual training s', 'Mean m', 'Median m', 'Mean episode CI95%'], [
            [row['metadata']['requested_minutes'], f"{row['metadata']['training_elapsed_seconds']:.3f}", f"{row['results']['mean_distance']:.2f}", f"{row['results']['median_distance']:.2f}", f"[{row['results']['summary']['distance']['ci95_low']:.2f}, {row['results']['summary']['distance']['ci95_high']:.2f}]"]
            for row in checkpoint_rows
        ])
        """),
        code("""
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout='constrained')
        axes[0].plot([r[0]/60 for r in training_curve], [r[1] for r in training_curve], color=COLORS['source'], linewidth=1.4)
        axes[0].set(title='Rolling last-100 training episodes: changing stochastic policies', ylabel='Training distance (nominal m)')
        times = [r['metadata']['training_elapsed_seconds']/60 for r in checkpoint_rows]
        means = [r['results']['mean_distance'] for r in checkpoint_rows]
        low = [m-r['results']['summary']['distance']['ci95_low'] for m,r in zip(means,checkpoint_rows)]
        high = [r['results']['summary']['distance']['ci95_high']-m for m,r in zip(means,checkpoint_rows)]
        axes[1].errorbar(times, means, yerr=[low,high], fmt='o-', color=COLORS['source'], capsize=4, label='Mean and episode bootstrap 95% CI')
        axes[1].plot(times, [r['results']['median_distance'] for r in checkpoint_rows], 's--', color=COLORS['shift'], label='Median')
        axes[1].set(title='Saved policies: same 20 validation seeds', xlabel='Actual training minutes', ylabel='Validation distance (nominal m)')
        for axis in axes:
            axis.set_xlim(left=0); axis.set_ylim(bottom=0); axis.grid(axis='y', alpha=.25)
        axes[1].legend(fontsize=9)
        plt.show()
        """),
        markdown(
            "### 4. Compare the final policy with earlier policies\nDifferences are paired by episode seed. The displayed intervals do not account for training-seed variability or multiple exploratory comparisons."
        ),
        code("""
        final = next(row for row in checkpoint_rows if row['metadata']['requested_minutes'] == 60)
        paired = [(row['metadata']['requested_minutes'], compare_paired(final['results'],row['results'])['distance_difference']) for row in checkpoint_rows if row is not final]
        show_table(['Earlier requested min', 'Final minus earlier mean m', 'Paired episode CI95%'], [[minute, f"{value['mean']:+.2f}", f"[{value['ci95_low']:.2f}, {value['ci95_high']:.2f}]"] for minute,value in paired])
        """),
        markdown(
            "### 5. Independently seeded one-hour reproduction\nThe primary remains fixed. Seed43 was a separate clean cold start, not a continuation or post-test replacement. Different shared-machine throughput is recorded; two seeds do not establish broad reliability."
        ),
        code("""
        reproduction = load_run(ROOT, 'db77b7cd-a11a-473a-8406-06d99b5de5ad')
        reproduction_checkpoints = load_run(ROOT, '153b8e02-dd2f-442f-9a48-6230d3f5c1cb')
        assert verify_run(ROOT,reproduction['run_id'])['valid']
        assert verify_run(ROOT,reproduction_checkpoints['run_id'])['valid']
        assert reproduction['parent_checkpoint'] is None and not reproduction['dirty_worktree']
        show_table(['Run','Seed','Actual training s','Transitions','Final mean / median m','Source SHA'],[
            [r['run_id'],r['seed'],f"{r['summary']['training_clock_seconds']:.3f}",r['environment_steps'],f"{r['summary']['mean_distance']:.2f} / {r['summary']['median_distance']:.2f}",r['git_sha']]
            for r in [primary,reproduction]
        ])
        show_table(['Requested min','Reproduction actual s','Mean / median m'],[
            [r['metadata']['requested_minutes'],f"{r['metadata']['training_elapsed_seconds']:.3f}",f"{r['results']['mean_distance']:.2f} / {r['results']['median_distance']:.2f}"]
            for r in sorted(reproduction_checkpoints['evaluation_results'],key=lambda r:r['metadata']['requested_minutes'])
        ])
        reproduction_pair = compare_paired(reproduction['evaluation_results'][-1]['results'],primary['evaluation_results'][-1]['results'])['distance_difference']
        print('Reproduction minus primary: conditional paired episode statistics',reproduction_pair)
        """),
        markdown("## Takeaways"),
        code("""
        increases = sum(checkpoint_rows[index]['results']['mean_distance'] > checkpoint_rows[index-1]['results']['mean_distance'] for index in range(1,len(checkpoint_rows)))
        intervals_cross_zero = sum(value['ci95_low'] <= 0 <= value['ci95_high'] for _,value in paired)
        display(Markdown(f"Observed mean increased at **{increases}/{len(checkpoint_rows)-1}** successive saved checkpoints. **{intervals_cross_zero}/{len(paired)}** final-versus-earlier paired intervals include zero. The completed seed43 reproduction supplies a second cold-start result, with final mean **{reproduction['summary']['mean_distance']:.2f}** nominal m; it does not establish universal training-seed reliability. No simulator result here establishes real-game control."))
        """),
    ]


def generalization_notebook():
    return [
        markdown("""
        # Vehicle/map generalization and adaptation
        ## Summary
        This notebook reads the fixed primary policy's four-condition generalization result. It also queries completed child policies for paired target improvement and source retention. Missing adaptation evidence stays missing; no inference or training occurs.
        """),
        code(
            SETUP.replace(
                "# ANALYSIS_IMPORTS",
                "from analyze_research import adaptation_comparisons\n\nfrom gradientclimb.experiments import list_runs, load_run, verify_run",
            )
        ),
        markdown("""
        ## Context & methods
        The default/train condition uses a fresh procedural seed set. Heavy is a synthetic vehicle profile; rough is a shifted synthetic terrain family. These are not commercial-game vehicles or maps.
        ### Key assumptions
        Generalization uses the same 20 seeds 20000–20019 across conditions. The 60-second horizon is finite. Episode uncertainty is conditional on the fixed policy; one parent/child training seed cannot establish adaptation reliability. Reuse of the same test seeds supports pairing but does not produce independent untouched test sets for later decisions.
        ## Data
        ### 1. Verify the generalization record and checkpoint lineage
        """),
        code("""
        generalization = load_run(ROOT, GENERALIZATION_EVALUATION)
        assert verify_run(ROOT, GENERALIZATION_EVALUATION)['valid']
        assert generalization['parent_checkpoint'] == primary['checkpoint_hash']
        conditions = ['in_distribution','new_map','new_vehicle','new_vehicle_and_map']
        results = {row['metadata']['condition']:row for row in generalization['evaluation_results']}
        assert set(results) == set(conditions)
        assert all(row['results']['seeds'] == list(range(20000,20020)) for row in results.values())
        print('Evaluation run:', GENERALIZATION_EVALUATION)
        print('Checkpoint SHA-256:', primary['checkpoint_hash'])
        print('Record SHA-256:', hashlib.sha256((ROOT/'runs'/GENERALIZATION_EVALUATION/'run.json').read_bytes()).hexdigest())
        """),
        markdown("## Results\n### 2. Compare distance and survival outcomes"),
        code("""
        show_table(['Condition', 'Mean / median nominal m', 'Mean survival s', 'Termination fractions'], [
            [condition, f"{results[condition]['results']['mean_distance']:.2f} / {results[condition]['results']['median_distance']:.2f}", f"{results[condition]['results']['summary']['survival_seconds']['mean']:.2f}", results[condition]['results']['summary']['failure_rates']]
            for condition in conditions
        ])
        """),
        code("""
        labels = ['Default / train','Default / rough','Heavy / train','Heavy / rough']
        distances = [results[c]['results']['summary']['distance'] for c in conditions]
        positions = list(range(4))
        fig, axis = plt.subplots(figsize=(9,4.7), layout='constrained')
        bars = axis.bar(positions, [d['mean'] for d in distances], color=[COLORS['source'],COLORS['shift'],COLORS['source'],COLORS['shift']], edgecolor='#192b35', linewidth=.7)
        for index, bar in enumerate(bars):
            if index%2:
                bar.set_hatch('//')
        axis.errorbar(positions,[d['mean'] for d in distances],yerr=[[d['mean']-d['ci95_low'] for d in distances],[d['ci95_high']-d['mean'] for d in distances]],fmt='none',ecolor='#192b35',capsize=4)
        axis.set_xticks(positions,labels)
        axis.set(ylabel='Mean distance (nominal m)',title='Primary final policy: 20 paired generalization seeds')
        axis.set_ylim(0,max(d['ci95_high'] for d in distances)*1.13); axis.grid(axis='y',alpha=.25)
        for index, distance in enumerate(distances):
            axis.annotate(f"{distance['mean']:.1f}", (index,distance['ci95_high']), xytext=(0,6), textcoords='offset points', ha='center', fontsize=10)
        plt.show()
        """),
        markdown(
            "### 3. Query actual completed adaptation evidence\nThe reusable report pairing helper requires matching checkpoints, scenario, horizon, simulator/calibration version, deterministic setting and episode seeds. Positive changes favor the child. Default/train measures retention on the original source condition. Condition names refer to the original parent; rough terrain was trained during the adapted child's exposure."
        ),
        code("""
        records = list_runs(ROOT)
        completed = [r for r in records if r['status']=='completed']
        evaluations = [dict(row,run_id=r['run_id']) for r in completed for row in r['evaluation_results']]
        comparisons = [row for row in adaptation_comparisons(completed,evaluations) if row['parent_training_run']==PRIMARY]
        if comparisons:
            show_table(['Child run','Condition','Before / after mean m','Paired change m','Paired episode CI95%'],[
                [row['child_training_run'],row['condition'],f"{row['before_mean']:.2f} / {row['after_mean']:.2f}",f"{row['difference_after_minus_before']['distance_difference']['mean']:+.2f}",f"[{row['difference_after_minus_before']['distance_difference']['ci95_low']:.2f}, {row['difference_after_minus_before']['distance_difference']['ci95_high']:.2f}]"]
                for row in comparisons
            ])
            target = next(row for row in comparisons if row['child_training_run']=='24eacdf3-cff5-430f-8978-b2e2bd34f396' and row['condition']=='new_vehicle_and_map')
            retention = next(row for row in comparisons if row['child_training_run']==target['child_training_run'] and row['condition']=='in_distribution')
            display(Markdown(f"The completed heavy/rough fine-tuning improved its target mean from **{target['before_mean']:.2f}** to **{target['after_mean']:.2f}** nominal m, while original source mean declined from **{retention['before_mean']:.2f}** to **{retention['after_mean']:.2f}**. This is a measured target/retention tradeoff for one child seed. Both ten-minute children load the original parent independently; no matched cold-start heavy/rough baseline establishes adaptation speed."))
        else:
            print('No matching completed parent/child generalization pairs are recorded yet. Adaptation and forgetting are not yet measured.')
        """),
        markdown("## Takeaways"),
        code("""
        source = results['in_distribution']['results']['mean_distance']
        rough = results['new_map']['results']['mean_distance']
        heavy = results['new_vehicle']['results']['mean_distance']
        crash = results['new_map']['results']['summary']['failure_rates'].get('crash',0)
        display(Markdown(f"On this fixed 20-seed result, rough terrain reduced mean distance from **{source:.2f}** to **{rough:.2f}** nominal m, with crashes in **{crash:.0%}** of episodes. The heavy vehicle on training terrain averaged **{heavy:.2f}** nominal m. This is evidence of a terrain-shift weakness in the surrogate, not a measured commercial-game transfer ratio. Longer unexecuted adaptation budgets remain missing."))
        """),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    directory = PROJECT / "notebooks"
    preview = PROJECT / "artifacts/notebooks"
    directory.mkdir(exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    runtime = preview / "runtime"
    for variable, leaf in [
        ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
        ("IPYTHONDIR", "ipython"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("JUPYTER_CONFIG_DIR", "jupyter-config"),
    ]:
        path = runtime / leaf
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["JUPYTER_PATH"] = str(runtime)
    kernel = runtime / "kernels/gradientclimb-local"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "GradientClimb local venv",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    for name, factory in [
        ("checkpoint_learning", checkpoint_notebook),
        ("generalization_adaptation", generalization_notebook),
    ]:
        notebook = nbformat.v4.new_notebook(
            cells=factory(),
            metadata={
                "kernelspec": {
                    "name": "gradientclimb-local",
                    "display_name": "GradientClimb local venv",
                    "language": "python",
                },
                "language_info": {"name": "python", "version": sys.version.split()[0]},
            },
        )
        for index, cell in enumerate(notebook.cells):
            cell.id = f"{name}-{index:02d}"
        nbformat.validate(notebook)
        if args.execute:
            from nbclient import NotebookClient
            from nbconvert import HTMLExporter

            NotebookClient(
                notebook,
                timeout=120,
                kernel_name="gradientclimb-local",
                resources={"metadata": {"path": str(PROJECT)}},
            ).execute()
            html_text, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
            (preview / f"{name}.html").write_text(html_text, encoding="utf-8")
            figure_count = 0
            for cell in notebook.cells:
                for output in cell.get("outputs", []):
                    png = output.get("data", {}).get("image/png")
                    if png:
                        figure_count += 1
                        (preview / f"{name}-figure{figure_count}.png").write_bytes(
                            base64.b64decode(png)
                        )
            print(
                json.dumps({"notebook": name, "executed": True, "figures": figure_count}),
                flush=True,
            )
        notebook.metadata.kernelspec = {
            "name": "python3",
            "display_name": "Python 3 (.venv)",
            "language": "python",
        }
        nbformat.validate(notebook)
        nbformat.write(notebook, directory / f"{name}.ipynb")


if __name__ == "__main__":
    main()
