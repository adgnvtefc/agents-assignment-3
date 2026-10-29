"""Standard diagnostic plots and summary tables for the report.

Reads runs/*.jsonl (and eval_results/p5_*.json for the Part 5 table) and
writes to plots/: one full multi-panel diagnostic figure per report section
(for the appendix), two focused panels per section (key_*.png, for the body),
and tables.md.

    uv run --with matplotlib python scripts/make_plots.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "plots"

# One fixed color per run across every figure. Each figure's set was checked
# with the dataviz palette validator (CVD and normal-vision separation).
RUN_INFO = {
    "p2a_number_search_reinforce": ("p2a  REINFORCE, G=1", "#eb6834"),
    "p2b_number_search_grpo": ("p2b  GRPO, G=8 (sync)", "#2a78d6"),
    "p41_number_search_grpo_g2": ("p41  GRPO, G=2", "#1baf7a"),
    "p3_number_search_async": ("p3  async GRPO, staleness 2", "#e34948"),
    "p43_number_search_async_stale4_iw": ("p43 iw  PPO, staleness 4", "#008300"),
    "p43_number_search_async_stale4_noiw": ("p43 noiw  weighted CE, staleness 4", "#e87ba4"),
    "p5_textcraft_grpo": ("p5  TextCraft async GRPO", "#6250d6"),
}

FIGURES = {
    "part2_reinforce_vs_grpo": ("Part 2: REINFORCE (p2a) vs. GRPO (p2b)",
                                ["p2a_number_search_reinforce", "p2b_number_search_grpo"]),
    "part3_async_grpo": ("Part 3: asynchronous GRPO (p3)", ["p3_number_search_async"]),
    "part4_1_group_size": ("Part 4.1: group size (G=1, 2, 8)",
                           ["p2a_number_search_reinforce", "p41_number_search_grpo_g2",
                            "p2b_number_search_grpo"]),
    "part4_2_sync_vs_async": ("Part 4.2: synchronous (p2b) vs. asynchronous (p3)",
                              ["p2b_number_search_grpo", "p3_number_search_async"]),
    "part4_3_importance_weighting": ("Part 4.3: importance weighting at staleness 4",
                                     ["p43_number_search_async_stale4_iw",
                                      "p43_number_search_async_stale4_noiw"]),
    "part5_textcraft": ("Part 5: TextCraft-Synth (p5)", ["p5_textcraft_grpo"]),
}

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})


def load(run: str) -> dict:
    train, evals, summary = [], [], {}
    for line in open(RUNS / f"{run}.jsonl"):
        r = json.loads(line)
        if any(k.startswith("summary/") for k in r):
            summary.update({k: v for k, v in r.items() if k.startswith("summary/")})
        elif "eval/success_rate" in r:
            evals.append(r)
        elif "train/success_rate" in r:
            train.append(r)
    return {"train": train, "eval": evals, "summary": summary}


def series(records: list[dict], key: str) -> tuple[list, list]:
    """(steps, values); a step without the key (e.g. a skipped update) is NaN,
    so the line breaks there instead of bridging the gap."""
    return [r["step"] for r in records], [
        float("nan") if r.get(key) is None else r[key] for r in records]


def train_pass_key(data: dict) -> str:
    return next(k for k in data["train"][0] if k.startswith("train/pass@"))


def diagnostic_figure(name: str, title: str, runs: list[str], data: dict) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    ax = dict(zip(["succ", "pass", "grad", "iwmean", "iwmin", "iwmax", "iwstd", "degen", "ent"],
                  axes.flat))
    for run in runs:
        d, (label, color) = data[run], RUN_INFO[run]
        tr = d["train"]
        short = label.split("  ")[0]
        # Small dots keep a lone step between two gaps (skipped updates) visible.
        solid = dict(color=color, lw=1.6, marker="o", ms=2)
        dashed = dict(color=color, lw=1.6, ls="--", marker="o", ms=4.5,
                      markeredgecolor="white", markeredgewidth=0.8)

        ax["succ"].plot(*series(tr, "train/success_rate"), **solid)
        ax["succ"].plot(*series(d["eval"], "eval/success_rate"), **dashed)

        pk = train_pass_key(d)
        ax["pass"].plot(*series(tr, pk), **solid, label=f"{short} {pk}")
        ax["pass"].plot(*series(d["eval"], "eval/pass@4"), **dashed,
                        label=f"{short} eval/pass@4")

        for key, k in [("grad", "train/grad_norm"), ("iwmean", "train/importance_weight_mean"),
                       ("iwmin", "train/importance_weight_min"),
                       ("iwmax", "train/importance_weight_max"),
                       ("iwstd", "train/importance_weight_std"), ("ent", "train/entropy_proxy")]:
            ax[key].plot(*series(tr, k), **solid)

        # REINFORCE has no group baseline: its logged 0 is meaningless, so it is not drawn.
        if not run.startswith("p2a"):
            ax["degen"].plot(*series(tr, "train/frac_degenerate_groups"), **solid)

    titles = {
        "succ": "success rate  (train/success_rate, eval/success_rate)",
        "pass": "pass@k  (exact keys in legend)",
        "grad": "train/grad_norm",
        "iwmean": "train/importance_weight_mean",
        "iwmin": "train/importance_weight_min",
        "iwmax": "train/importance_weight_max",
        "iwstd": "train/importance_weight_std",
        "degen": "train/frac_degenerate_groups"
                 + ("  (p2a: N/A)" if "p2a_number_search_reinforce" in runs else ""),
        "ent": "train/entropy_proxy",
    }
    for key, a in ax.items():
        a.set_title(titles[key], loc="left", color=INK)
        a.set_xlabel("training step")
        a.ticklabel_format(axis="y", useOffset=False)
    for key in ("succ", "pass", "degen"):
        ax[key].set_ylim(-0.03, 1.03)
    ax["iwmean"].axhline(1.0, color=INK_2, lw=0.8, ls=":")
    ax["pass"].legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9,
                      edgecolor="none")

    handles = [Line2D([], [], color=RUN_INFO[r][1], lw=2.5, label=RUN_INFO[r][0]) for r in runs]
    handles += [Line2D([], [], color=INK_2, lw=1.6, label="train (every step)"),
                Line2D([], [], color=INK_2, lw=1.6, ls="--", marker="o", ms=4.5,
                       markeredgecolor="white", label="eval (every 10 steps)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 5),
               frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=12, color=INK)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def minutes_at_step(d: dict) -> dict:
    """Cumulative training minutes (sum of step_time_s) at each step; step 0 is 0."""
    out, total = {0: 0.0}, 0.0
    for r in d["train"]:
        total += r.get("train/step_time_s") or 0.0
        out[r["step"]] = total / 60
    return out


def key_figure(name: str, panels: list, data: dict) -> None:
    """Two focused panels for the body of a report section.
    panels: [(title, xlabel, [(run, kind, key, label_suffix), ...]), ...] where kind is
    "train" / "eval" (vs. step), "eval_min" (eval vs. training minutes), or "min_dotted"
    (a train series drawn dotted, for a second statistic of the same run)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
    for a, (title, xlabel, lines) in zip(axes, panels):
        for run, kind, key, suffix in lines:
            short, color = RUN_INFO[run][0].split("  ")[0], RUN_INFO[run][1]
            d = data[run]
            if kind == "eval_min":
                mins = minutes_at_step(d)
                xs = [mins[r["step"]] for r in d["eval"]]
                ys = [r[key] for r in d["eval"]]
                a.plot(xs, ys, color=color, lw=1.8, marker="o", ms=4.5,
                       markeredgecolor="white", label=f"{short} {suffix}".strip())
                continue
            xs, ys = series(d["eval"] if kind == "eval" else d["train"], key)
            style = dict(color=color, lw=1.6, marker="o", ms=2)
            if kind == "eval":
                style.update(ls=":" if key == "eval/pass@4" else "--", ms=4.5,
                             markeredgecolor="white")
            if kind == "min_dotted":
                style.update(ls=":", lw=1.8)
            a.plot(xs, ys, label=f"{short} {suffix}".strip(), **style)
        a.set_title(title, loc="left", color=INK)
        a.set_xlabel(xlabel)
        a.ticklabel_format(axis="y", useOffset=False)
        a.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


P2A, P2B, P41, P3 = ("p2a_number_search_reinforce", "p2b_number_search_grpo",
                     "p41_number_search_grpo_g2", "p3_number_search_async")
IW, NOIW, P5 = ("p43_number_search_async_stale4_iw", "p43_number_search_async_stale4_noiw",
                "p5_textcraft_grpo")
STEP = "training step"
KEY_FIGURES = {
    "key_part2": [
        ("eval/success_rate", STEP, [(P2A, "eval", "eval/success_rate", ""),
                                     (P2B, "eval", "eval/success_rate", "")]),
        ("train/grad_norm", STEP, [(P2A, "train", "train/grad_norm", ""),
                                   (P2B, "train", "train/grad_norm", "")]),
    ],
    "key_part3": [
        ("importance weight tails: max (solid), min (dotted)", STEP,
         [(P2B, "train", "train/importance_weight_max", "max"),
          (P2B, "min_dotted", "train/importance_weight_min", "min"),
          (P3, "train", "train/importance_weight_max", "max"),
          (P3, "min_dotted", "train/importance_weight_min", "min")]),
        ("train/step_time_s (seconds)", STEP, [(P2B, "train", "train/step_time_s", ""),
                                               (P3, "train", "train/step_time_s", "")]),
    ],
    "key_part4_1": [
        ("eval/success_rate", STEP, [(P2A, "eval", "eval/success_rate", ""),
                                     (P41, "eval", "eval/success_rate", ""),
                                     (P2B, "eval", "eval/success_rate", "")]),
        ("train/frac_degenerate_groups (p2a: N/A)", STEP,
         [(P41, "train", "train/frac_degenerate_groups", ""),
          (P2B, "train", "train/frac_degenerate_groups", "")]),
    ],
    "key_part4_2": [
        ("eval/success_rate vs. step", STEP, [(P2B, "eval", "eval/success_rate", ""),
                                              (P3, "eval", "eval/success_rate", "")]),
        ("eval/success_rate vs. wall-clock", "training time (minutes)",
         [(P2B, "eval_min", "eval/success_rate", ""),
          (P3, "eval_min", "eval/success_rate", "")]),
    ],
    "key_part4_3": [
        ("eval/success_rate", STEP, [(IW, "eval", "eval/success_rate", ""),
                                     (NOIW, "eval", "eval/success_rate", "")]),
        ("importance weight tails: max (solid), min (dotted)", STEP,
         [(IW, "train", "train/importance_weight_max", "max"),
          (IW, "min_dotted", "train/importance_weight_min", "min"),
          (NOIW, "train", "train/importance_weight_max", "max"),
          (NOIW, "min_dotted", "train/importance_weight_min", "min")]),
    ],
    "key_part5": [
        ("eval/success_rate and eval/pass@4", STEP,
         [(P5, "eval", "eval/success_rate", "eval/success_rate"),
          (P5, "eval", "eval/pass@4", "eval/pass@4")]),
        ("train/entropy_proxy", STEP, [(P5, "train", "train/entropy_proxy", "")]),
    ],
}


def fmt(x, nd=3):
    return "N/A" if x is None else f"{x:.{nd}f}"


def tables(data: dict) -> str:
    lines = ["# Summary tables", "",
             "## Per-run summary", "",
             "| run | final eval/success_rate | final eval/pass@4 | step-0 eval/success_rate "
             "| mean step_time_s | median step_time_s | total train time (min) "
             "| mean frac_degenerate_groups | groups_consumed | groups_dropped_stale |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run, d in data.items():
        ev, tr, sm = d["eval"], d["train"], d["summary"]
        st = [r["train/step_time_s"] for r in tr if r.get("train/step_time_s") is not None]
        degen = None if run.startswith("p2a") else statistics.mean(
            r["train/frac_degenerate_groups"] for r in tr)
        lines.append(
            f"| {run} | {fmt(ev[-1]['eval/success_rate'])} (step {ev[-1]['step']}) "
            f"| {fmt(ev[-1]['eval/pass@4'])} | {fmt(ev[0]['eval/success_rate'])} "
            f"| {statistics.mean(st):.1f} | {statistics.median(st):.1f} | {sum(st) / 60:.1f} "
            f"| {fmt(degen)} | {sm.get('summary/groups_consumed', '-')} "
            f"| {sm.get('summary/groups_dropped_stale', '-')} |")

    lines += ["", "## Importance-weight statistics over training", "",
              "| run | mean of importance_weight_mean | min of importance_weight_min "
              "| max of importance_weight_max | mean of importance_weight_std "
              "| max of importance_weight_std |",
              "|---|---|---|---|---|---|"]
    for run, d in data.items():
        tr = d["train"]

        def col(k):
            return [r[k] for r in tr if r.get(k) is not None]

        lines.append(
            f"| {run} | {statistics.mean(col('train/importance_weight_mean')):.4f} "
            f"| {min(col('train/importance_weight_min')):.4f} "
            f"| {max(col('train/importance_weight_max')):.4f} "
            f"| {statistics.mean(col('train/importance_weight_std')):.4f} "
            f"| {max(col('train/importance_weight_std')):.4f} |")

    lines += ["", "## Part 5 final evaluation (k = 4)", "",
              "| model | split | success_rate | pass@4 | avg_steps | abandoned_episodes |",
              "|---|---|---|---|---|---|"]
    for model, mfile in [("fine-tuned (v50 adapter)", "finetuned"), ("base Qwen3.5-4B", "base")]:
        for split, sfile in [("medium (213 tasks)", "val"), ("hard (136 tasks)", "hard")]:
            p = ROOT / "eval_results" / f"p5_{mfile}_{sfile}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            lines.append(f"| {model} | {split} | {r['success_rate']:.3f} | {r['pass@4']:.3f} "
                         f"| {r['avg_steps']:.1f} | {r['abandoned_episodes']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = {run: load(run) for run in RUN_INFO}
    for name, (title, runs) in FIGURES.items():
        diagnostic_figure(name, title, runs, data)
    for name, panels in KEY_FIGURES.items():
        key_figure(name, panels, data)
    md = tables(data)
    (OUT / "tables.md").write_text(md)
    print(md)
    print("figures:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
