# Reproduction report (quick_smoke_20261121_acc)

**Quick mode (1 rep, 1 round / 3 chunks): a smoke test of the commands, not a result.**

## accuracy-olmoe (RESULTS 1ze; README.md, accuracy paragraph; docs/models.md)

| run | published | new | verdict |
| --- | --- | --- | --- |
| olmoe_mj_bf16 | ppl 11.540 +- 0.251 | ppl 8.859 +- 1.006 | ran (quick: not compared) |
| olmoe_mj_q4_kld | ppl 12.004 +- 0.258, KL 0.1592 | ppl 9.395 +- 1.068, KL 0.1692 | ran (quick: not compared) |
| olmoe_mlx-lm_q4_kld | ppl 11.987 +- 0.257, KL 0.1589 | ppl 9.410 +- 1.070, KL 0.1683 | ran (quick: not compared) |
| olmoe_llamacpp_q4km | ppl 12.000 +- 0.263 | ppl 9.487 +- 1.103 | ran (quick: not compared) |

