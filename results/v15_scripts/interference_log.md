# Other load on the machine during the 2026-11-21/22 runs (for judging blocks)

| when | what | effect to expect |
| --- | --- | --- |
| 2026-11-22 ~00:15 to ~01:05 | another agent session active (Claude app ~1.3 cores; swap up to 3.8 GB) | v15 first gpt-oss pass (`v15_gptoss`): treat as noisy |
| 02:10:10-02:13:03, 02:13:56-02:14:33 | our own offline simulations, one core, nice 19 (RESULTS 1zn) | overlaps `v15q_gptoss` r2 mj-a8w |
| from ~02:11 (seen 02:16; up to 50 min by its timeout) | another session's verifier, 2 Python workers at ~96% CPU each, negligible memory | overlaps `v15q_gptoss` r2 mj-a8w (flagged for swap anyway) and r2 mj-x, and the start of `v15q_dsv2lite` |

Blocks inside these windows are not used for claims without a rerun on a quiet machine.
| 02:17:45-02:20:53 | another session's LP job (pid 29011), ~0.9 core, 440 MB | overlaps `v15q_gptoss` r2 mj-x |
| ~02:20 onward (seen 02:20:57; minutes) | macOS XProtect Remediator scan, ~1.5 cores (system service, left alone) | end of `v15q_gptoss` r2 mj-x, start of `v15q_dsv2lite` |
| from 02:23 | the other agent session reports all its compute stopped and held until told the timed runs are done | machine clear of other agents from here |
