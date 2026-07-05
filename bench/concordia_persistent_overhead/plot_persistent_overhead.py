#!/usr/bin/env python3
import argparse
import csv
import io
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
ORANGE = {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]

    use_chart_theme()
    overhead = read_csv_with_comments(Path(args.overhead_csv))
    delta_sections = read_delta_log(Path(args.delta_log))

    write_persistent_overhead(overhead, out_dir, formats)
    write_delta_rerun(delta_sections, out_dir, formats)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Concordia evaluation figures from measured CSV/log artifacts.")
    parser.add_argument("--overhead-csv", required=True, help="CSV emitted by concordia_persistent_overhead_bench")
    parser.add_argument("--delta-log", required=True, help="Log emitted by bench/concordia_delta_checkpoint")
    parser.add_argument("--out-dir", required=True, help="Directory for generated figure files")
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated output formats")
    return parser.parse_args()


def read_csv_with_comments(path: Path) -> pd.DataFrame:
    lines = []
    in_table = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("worker_blocks,"):
            in_table = True
        if in_table:
            lines.append(line)
    if not lines:
        raise ValueError(f"{path} contains no worker_blocks CSV table")
    return pd.read_csv(io.StringIO("\n".join(lines)))


def read_delta_log(path: Path) -> dict[str, pd.DataFrame]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = line[1:].strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)

    frames = {}
    for name, rows in sections.items():
        if rows:
            frames[name] = pd.read_csv(io.StringIO("\n".join(rows)))
    required = {"table5_sparse_delta_checkpoint", "table6_dirty_scaling_256mb"}
    missing = required.difference(frames)
    if missing:
        raise ValueError(f"{path} missing sections: {sorted(missing)}")
    return frames


def write_persistent_overhead(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    required = {
        "worker_blocks",
        "theoretical_sm_pct",
        "copy_overhead_pct",
        "compute_overhead_pct",
        "copy_ms",
        "compute_ms",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"overhead CSV missing columns: {sorted(missing)}")

    plot_df = df.loc[df["worker_blocks"] > 0].copy()
    long = plot_df.melt(
        id_vars=["worker_blocks", "theoretical_sm_pct"],
        value_vars=["copy_overhead_pct", "compute_overhead_pct"],
        var_name="workload",
        value_name="overhead_pct",
    )
    long["workload"] = long["workload"].map(
        {
            "copy_overhead_pct": "Memory kernel",
            "compute_overhead_pct": "FMA kernel",
        }
    )

    baseline = df.loc[df["worker_blocks"] == 0].iloc[0]
    subtitle = (
        f"Resident worker blocks share one CUDA context; baseline copy={baseline['copy_ms']:.3f} ms "
        f"and FMA={baseline['compute_ms']:.3f} ms."
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    palette = {"Memory kernel": BLUE["base"], "FMA kernel": ORANGE["base"]}
    sns.lineplot(
        data=long,
        x="worker_blocks",
        y="overhead_pct",
        hue="workload",
        style="workload",
        markers=True,
        dashes=False,
        palette=palette,
        linewidth=1.25,
        markersize=6,
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color=TOKENS["ink"], linewidth=0.9)
    ax.set_xlabel("Resident Worker Blocks")
    ax.set_ylabel("Runtime Overhead")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(plot_df["worker_blocks"].tolist())
    ax.set_xlim(plot_df["worker_blocks"].min() - 0.35, plot_df["worker_blocks"].max() + 0.75)
    ax.grid(axis="x", visible=False)

    for workload, color in palette.items():
        endpoint = long.loc[long["workload"] == workload].sort_values("worker_blocks").iloc[-1]
        ax.text(
            endpoint["worker_blocks"] + 0.18,
            endpoint["overhead_pct"],
            workload,
            va="center",
            ha="left",
            fontsize=8,
            color=color,
        )

    for _, row in plot_df.iterrows():
        ax.annotate(
            f"{row['theoretical_sm_pct']:.2f}% SM",
            xy=(row["worker_blocks"], max(row["copy_overhead_pct"], row["compute_overhead_pct"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=TOKENS["muted"],
        )

    add_chart_header(
        fig,
        ax,
        "Persistent-worker overhead scales with reserved blocks",
        subtitle,
    )
    export(fig, out_dir / "concordia_persistent_overhead_ablation", formats)


def write_delta_rerun(sections: dict[str, pd.DataFrame], out_dir: Path, formats: list[str]) -> None:
    sparse = sections["table5_sparse_delta_checkpoint"].copy()
    dirty = sections["table6_dirty_scaling_256mb"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    sns.barplot(
        data=sparse,
        x="region_mb",
        y="speedup",
        ax=axes[0],
        color=BLUE["base"],
        edgecolor=BLUE["dark"],
        linewidth=1.0,
    )
    axes[0].set_xlabel("Region Size")
    axes[0].set_ylabel("CPU/GPU Delta Speedup")
    axes[0].set_xticks(range(len(sparse)))
    axes[0].set_xticklabels([f"{int(v)} MB" for v in sparse["region_mb"]])
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}x"))
    axes[0].grid(axis="x", visible=False)

    sns.barplot(
        data=dirty,
        x="requested_dirty_pages",
        y="gpu_total_ms",
        ax=axes[1],
        color=OLIVE["base"],
        edgecolor=OLIVE["dark"],
        linewidth=1.0,
    )
    axes[1].set_xlabel("Dirty Pages at 256 MB")
    axes[1].set_ylabel("GPU Delta Time (ms)")
    axes[1].grid(axis="x", visible=False)

    for ax in axes:
        sns.despine(ax=ax)

    add_chart_header(
        fig,
        axes[0],
        "Artifact rerun confirms GPU-side delta advantage",
        "cuda-oxide scanner on RTX PRO 6000; speedup is lower than the paper's optimized JIT handler but preserves the scaling trend.",
    )
    fig.subplots_adjust(wspace=0.34)
    export(fig, out_dir / "concordia_artifact_delta_rerun", formats)


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title.strip(), width=78, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=112, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    fig.subplots_adjust(top=max(0.62, 0.84 - 0.045 * (title_lines - 1) - 0.032 * (subtitle_lines - 1)))
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=12, fontweight="semibold", color=TOKENS["ink"], linespacing=1.08)
    fig.text(left, 0.925 - 0.045 * (title_lines - 1), subtitle, ha="left", va="top", fontsize=8.2, color=TOKENS["muted"], linespacing=1.18)
    sns.despine(ax=ax)


def export(fig, stem: Path, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
