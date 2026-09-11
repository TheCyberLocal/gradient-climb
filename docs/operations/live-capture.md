# Live capture measurements

The initial native benchmarks ran on 2026-09-11 against the visible Google Play Games window. The user confirmed Right Arrow for gas and Left Arrow for brake. The physical client was 2581 × 1449 pixels on the Intel-connected 3840 × 2160 primary display. A normalized 1034 × 581 observation retains the wrapper sidebar and title bar; profile coordinates refer to that normalized image.

The one-hour CPU surrogate training session was active during these measurements. Runs were sequential, with different menu/paused scenes, no repetitions, and no temperature control. These are engineering measurements, not a controlled statistical comparison.

| Backend and recording | Frames | Measured observations/s | Mean capture call ms | Canonical run |
| --- | ---: | ---: | ---: | --- |
| MSS, PNG every frame | 60 | 1.83 | 149.55 | e003676b-714f-4555-92c8-801d9ba6283a |
| Pillow, no recording | 30 | 4.50 | 220.80 | a81a6a9c-2595-459b-9468-443b436b65fb |
| DXcam, no recording | 60 | 11.76 | 83.52 | d6a08878-ef13-4924-987d-1416a9e0cfbd |
| MSS, no recording | 30 | 6.62 | 149.61 | f01e0cd8-735d-4dae-a4d0-c9ff3e77b027 |

The recorded MSS run missed 59 of 60 requested 30 Hz scheduling slots. This is not a measurement of dropped compositor frames. PNG encoding and writing count in its end-to-end duration. Its capture-call metric excludes encoding.

DXcam uses ordinary desktop duplication and reads rendered pixels. The adapter initially supports device 0/output 0 at desktop origin (0, 0), checks the selected game identity/foreground before and after capture, and rejects rectangles crossing output bounds. It neither focuses windows nor reads game memory. `new_frame_only=False` may return an unchanged display image. Source presentation timestamps and dropped/new-frame counts are not exposed through this adapter; capture timestamps are API interval midpoints, not presentation timestamps. Do not interpret capture rate as game render rate or true display-to-policy latency.

Reproduce a bounded comparison with `python -m gradientclimb capture --backend dxcam --frames 60`. Run it while the verified game remains in the foreground. Desktop-isolated execution cannot enumerate the real game and must fail without sending input.

The native normalized paused frame failed the original tool-screenshot template threshold. Preserve that rejection as a domain/alignment failure. A new native playing reference may be used for supervised probe construction, but self-matching it does not establish held-out state accuracy. No template alone authorizes unknown menu or advertisement clicks.

Source: [DXcam documentation](https://github.com/ra1nty/DXcam). Exact dependency versions are recorded in `requirements-lock.txt`; original captures stay in ignored local artifact storage.
