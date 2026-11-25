"""Tweet charts (no engine name). Every number below is copied from a run's summary (RESULTS 1zi, 1zk, 1zl)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a85", "#e6e5e1"
OURS, RIVAL = "#2a78d6", "#85837e"
FOOT = "Base M4 MacBook Pro, 16 GB · mean of alternating rounds, bars = 95% confidence interval · higher is better"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def hbars(ax, names, vals, lo, hi, colors, fmt="{:.1f}", note=None, xmax=None):
    y = list(range(len(names)))[::-1]
    err = [[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURF, linewidth=2)
    ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=INK2, elinewidth=1.4, capsize=4)
    ax.set_yticks(y, names)
    for yy, v, h in zip(y, vals, hi):
        ax.text(h + (xmax or max(hi)) * 0.03, yy, fmt.format(v), va="center", fontsize=13, color=INK)
    if xmax:
        ax.set_xlim(0, xmax)
    if note:
        ax.text(0, -0.75, note, transform=ax.get_yaxis_transform(), fontsize=11, color=MUTED, va="top")


def finish(fig, headline, path, foot=FOOT):
    fig.suptitle(headline, x=0.02, ha="left", fontsize=19, fontweight="bold", color=INK, y=0.98)
    fig.text(0.02, 0.015, foot, fontsize=10.5, color=MUTED)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


# 1. Qwen3.6-35B-A3B, bigger than RAM (RESULTS 1zl reproduction; ours n 10, llama.cpp n 3)
fig, axs = plt.subplots(2, 2, figsize=(12, 6.75))
fig.subplots_adjust(left=0.13, right=0.96, top=0.82, bottom=0.12, hspace=0.75, wspace=0.35)
cells = [("Decode, 1-token context (tok/s)", 22.84, 20.93, 24.75, 4.42, 4.04, 4.80),
         ("Decode, 2,048-token context (tok/s)", 20.30, 19.34, 21.26, 4.77, 1.43, 8.11),
         ("Prefill, 512-token prompt (tok/s)", 169.73, 166.98, 172.47, 16.36, 15.49, 17.23),
         ("Prefill, 2,048-token prompt (tok/s)", 278.41, 276.81, 280.00, 13.92, 10.44, 17.40)]
for ax, (t, o, ol, oh, l, ll, lh) in zip(axs.flat, cells):
    style(ax, t)
    hbars(ax, ["ours", "llama.cpp"], [o, l], [ol, ll], [oh, lh], [OURS, RIVAL], xmax=oh * 1.22)
    ax.text(0.98, 0.5, f"{o / l:.1f}x", transform=ax.transAxes, ha="right", va="center", fontsize=22,
            fontweight="bold", color=OURS)
finish(fig, "Qwen3.6-35B-A3B (20.4 GB) on a 16 GB Mac: experts streamed from the SSD", f"{OUT}/1_qwen36_bigger_than_ram.png",
       FOOT + "\nllama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40); mlx-lm cannot load this model on 16 GB")

# 2. gpt-oss-20b decode and GPU energy (RESULTS 1zk; ours = 8-bit attention, MXFP4 experts)
fig, axs = plt.subplots(1, 3, figsize=(12, 6.75))
fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.17, wspace=0.55)
eng = ["ours", "llama.cpp", "LM Studio", "Ollama"]
col = [OURS, RIVAL, RIVAL, RIVAL]
style(axs[0], "Decode, 1-token context\n(tok/s)")
hbars(axs[0], eng, [39.80, 36.95, 35.76, 35.50], [39.50, 36.86, 35.35, 35.31], [40.11, 37.03, 36.17, 35.69], col,
      xmax=48)
style(axs[1], "Decode, 2,048-token context\n(tok/s)")
hbars(axs[1], eng, [38.60, 35.09, 33.48, 34.24], [38.50, 33.66, 33.24, 33.96], [38.71, 36.52, 33.71, 34.51], col,
      xmax=48)
style(axs[2], "GPU energy per generated\ntoken (mJ, lower is better)")
e3 = ["ours", "LM Studio", "Ollama"]
hbars(axs[2], e3, [267, 373, 368], [267, 373, 368], [267, 373, 368], [OURS, RIVAL, RIVAL], fmt="{:.0f}", xmax=450)
axs[2].text(0, -0.75, "llama.cpp: not metered yet", transform=axs[2].get_yaxis_transform(), fontsize=11, color=MUTED,
            va="top")
finish(fig, "gpt-oss-20b on a 16 GB Mac: faster decode, less GPU energy", f"{OUT}/2_gptoss_decode_energy.png",
       FOOT + "\nours: MXFP4 experts, 8-bit attention weights (the GGUF's attention is 8-bit too); rivals: ggml-org MXFP4 GGUF;"
       " energy = GPU rail, 1-token context")

# 3. Long context (RESULTS 1zi)
fig, axs = plt.subplots(1, 3, figsize=(12, 6.75))
fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.2, wspace=0.55)
style(axs[0], "gpt-oss-20b, 16k prompt:\nprefill (tok/s)")
hbars(axs[0], ["ours", "llama.cpp"], [355.65, 275.83], [352.02, 274.56], [359.29, 277.10], [OURS, RIVAL], xmax=460)
style(axs[1], "gpt-oss-20b, decode at 16k\ncontext (tok/s)")
hbars(axs[1], ["ours", "llama.cpp"], [31.37, 27.89], [30.86, 26.79], [31.87, 28.99], [OURS, RIVAL], xmax=40)
style(axs[2], "DeepSeek-V2-Lite,\nprefill (tok/s)")
y = [3, 2, 1, 0]
names = ["ours, 8k", "llama.cpp, 8k", "ours, 16k", "llama.cpp, 16k"]
vals = [457.17, 0, 309.48, 0]
axs[2].barh(y, vals, height=0.62, color=[OURS, RIVAL, OURS, RIVAL], edgecolor=SURF, linewidth=2)
axs[2].errorbar([457.17, 309.48], [3, 1], xerr=[[457.17 - 444.42, 309.48 - 308.25], [469.91 - 457.17, 310.71 - 309.48]],
                fmt="none", ecolor=INK2, elinewidth=1.4, capsize=4)
axs[2].set_yticks(y, names)
axs[2].set_xlim(0, 600)
for yy, v in ((3, 457.17), (1, 309.48)):
    axs[2].text(v + 25, yy, f"{v:.1f}", va="center", fontsize=13, color=INK)
for yy in (2, 0):
    axs[2].text(8, yy, "out of memory", va="center", fontsize=12, color=INK2, style="italic")
finish(fig, "Long context on a 16 GB Mac", f"{OUT}/3_long_context.png",
       FOOT + "\nllama.cpp: gpt-oss fits at 16k only with batch = ubatch 128 (mean ± sd, n 3; f16 KV, flash attention);"
       " DeepSeek-V2-Lite out of Metal memory\nat batch 512, 256 and 128 at 8k and 16k. Ours, DeepSeek-V2-Lite 16k: 512-token prefill chunks")
print("ok")
