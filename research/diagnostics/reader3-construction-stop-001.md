The reader construction sequence stopped at the user's request after two completed evidence publications from clean source `1c80c3265a2a296db384d9b3ec44de87fa915553`: declaration `02886e4c-b804-4c0d-a619-9eda13b01ee9` and annotation `ac65bbd6-1c8e-4b7a-a57f-23fc1efa895d`. The builder and freeze were each invoked **zero times**. No candidate was created, and no new capture, inference, training, qualification or evaluated episode occurred. One of the original three native reliability sessions remains available; these offline publications do not reset or consume that budget.

The one published annotation says **269** for the full right-side result-distance field of native run `1c263031-ac29-4dbc-8fa5-864a2473116a`, attempt 8, retained `terminal-024` frame. It was explicitly reviewed as known-exposure construction evidence, with the exact first exposure and manual review costs unknown. Its canonical endpoint identity prevents treating another animation frame as a second endpoint. The original native endpoint remains `success_unscored` with `distance: null`. The label establishes neither blinded truth nor independent reader accuracy. Its session ID is an audit grouping of an older run whose original session ID was absent.

The declaration sealed at 17:36:26.709472 UTC and the annotation publication sealed at 17:37:59.103805 UTC on 12 September 2026. Their canonical statuses are completed. The annotation journal ends with `annotation_publication_completed`; its exact annotation and index are registered and sealed. The historical summary field `publication_complete: false` is a stale conservative pre-seal marker that was never promoted. Completion requires the owning completed canonical record, valid seal and exact registered annotation. The marker is retained as a metadata limitation; no sealed record was edited.

| Measured component | Declaration | Annotation publication |
| --- | ---: | ---: |
| Recorder wall seconds | 0.511750 | 13.213932 |
| Resource sample span seconds | 0.187652 | 12.961140 |
| Recorder-process sampled CPU core seconds | 0.015625 | 12.453125 |
| CPU coverage | 70.2853% | 99.5892% |
| Process RSS mean / sampled peak, bytes | 70,739,968 / 70,742,016 | 71,344,128 / 76,767,232 |
| Host RAM mean / sampled peak, bytes | 35,881,058,304 / 35,890,769,920 | 35,649,062,912 / 35,652,149,248 |
| Device VRAM mean / sampled peak, bytes | 844,103,680 / 844,103,680 | 844,103,680 / 844,103,680 |
| Device GPU utilization mean / sampled peak | 23.5% / 24% | 26% / 26% |
| Device GPU utilization-equivalent seconds | 0.029268 | 3.357347 |
| Device GPU / VRAM coverage | 66.3697% | 99.6276% |

Each resource record has only two samples. Process RSS coverage is 70.2798% and 99.5892%; host RAM coverage is 70.2964% and 99.5893%. Full exact values and uncovered intervals appear in the [numeric receipt](reader3-construction-stop-001.json). Means interpolate valid adjacent samples; peaks are sampled peaks and can miss higher unsampled use. CPU covers recorder-process threads and excludes child processes. GPU utilization and VRAM include all device processes and cannot be attributed to this annotation or a policy; energy was not measured.

The annotation operation journal separately measures 13.167488 seconds and 12.546875 CPU core seconds from publisher entry through its final event, including validation, hash reads and JSON publication. This overlaps recorder wall and resource measurements. **Do not add these clocks or CPU totals, or sum cumulative journal rows.** Recorder resources run from the initial resource sample to final sample completion. Imports, initial parsing/validation or preflight where applicable, later sealing and manual review lie outside the reported component boundaries. Full command cost is unknown; these figures are not a speed benchmark.

The [preparatory prior-cost report](reader3-construction-prior-costs-001.json) remains the inherited-cost reference: 24 acquisition or bank/anchor construction rows plus two explicit design-feedback rows, for 26 unique canonical rows, and six noncanonical inherited sources with unknown costs. It preserves each run's historical recorder wall, steps, episodes and available resource scopes. First viewing, manual labeling, engineering and complete causal allocation remain unknown; there is no defensible aggregate inherited total or independent session count. Adjacent studies with no established inherited edge are excluded.

The numeric receipt contains source, seal, protocol, annotation, journal, log and prior-report hashes. This review read metadata and logs only; it opened no images and performed no inference, tests, collection or fitting. The remaining reader qualification and rapid-learning goals are future work, not results of these publications.
