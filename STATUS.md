# Status

| Phase | Name | Status | Date | Notes |
|---|---|---|---|---|
| 0 | Setup and data access | done | 2026-10-25 | 11 tests pass; camera OK, Kaggle OK; asl-stats done (median 22 frames, p90 92); vocab.csv reviewed by user; user confirmed GIF look (face + both hands added on request) |
| 1 | Camera and live landmarks | done | 2026-10-26 | 34 tests pass; 13 recordings (hello 5, please 3, mom 5); FPS median 16.7 over the log but today 9-14 (throttling) - re-check in Phase 4; no mirroring; 640x480; Holistic complexity 1 |
| 2 | Features | done | 2026-10-26 | FEATURES_VERSION 1, F=542, T_FIX=64 (param); extract ~1 ms/clip; aspect 3:4 + 3 participant overrides (asl-check); trim_idle for live only (cuts 9.4% of dataset hand frames); reference-clip rule; GIFs at real speed |
| 3 | Training on Kaggle | in progress | 2026-10-26 | smoke test on the user's recordings: DEFERRED until the user re-records to their satisfaction (preliminary run on the 2026-10-26 19:2x recordings) |
| 4 | App shell and Single word mode | not started | | |
| 5 | Sentence mode | not started | | |
| 6 | Learn mode | not started | | |
| 7 | Translations and Armenian audio | not started | | |
| 8 | Packaging and defense kit | not started | | |
