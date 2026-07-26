"""Render the benchmark figure used at the top of README.md.

  left   the LUNA16 FROC operating curve, read from results/metrics/froc_subset0.json,
         so this panel cannot drift from the scored run
  right  the three headline metrics as point estimates with their stated uncertainty.
         Only the CPM row comes from that JSON. Dice and AUROC are constants below,
         taken from the run logs, because those two evaluations wrote no metrics file.
         If that changes, read them here instead of editing the numbers by hand.

Writes a light and a dark render so the README can serve the right one per theme:

    python scripts/plot_benchmarks.py --out docs/img

Only needed to regenerate the figure; matplotlib is not a runtime dependency of
the pipeline itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FROC_JSON = REPO_ROOT / "results" / "metrics" / "froc_subset0.json"

# Two deliberately-chosen themes rather than one auto-inverted. Values are GitHub's own
# surface/ink/accent steps, so the figure sits on the README background it is served against.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "accent": "#0969da",
        "interval": "#8c959f",
    },
    "dark": {
        "surface": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "grid": "#3d444d",
        "accent": "#58a6ff",
        "interval": "#6e7681",
    },
}


def _relative_luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    parts = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
    return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio. Computed rather than eyeballed."""
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    lo, hi = sorted((a, b))
    return (hi + 0.05) / (lo + 0.05)


def _headline_metrics(froc: dict) -> list[dict]:
    """The three stage metrics. Each carries the KIND of its interval, because they differ."""
    return [
        {
            "label": "Detection\nFROC / CPM",
            "value": froc["cpm"],
            "low": froc["ci95"][0],
            "high": froc["ci95"][1],
            "note": f"95% CI, {froc['n_scans']} scans, out of fold",
        },
        {
            # Round-1 Dice: mean 0.695, median 0.728, observed range 0.31 to 0.86.
            "label": "Segmentation\nDice",
            "value": 0.695,
            "low": 0.31,
            "high": 0.86,
            "note": "observed range, n=15 lesions",
        },
        {
            "label": "Malignancy\nAUROC",
            "value": 0.876,
            "low": 0.876 - 0.045,
            "high": 0.876 + 0.045,
            "note": "SD across folds, n=1353 nodules",
        },
    ]


def _panel_froc(ax, froc: dict, t: dict) -> None:
    points = sorted((float(k), v) for k, v in froc["sensitivities"].items())
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.set_xscale("log", base=2)
    ax.plot(xs, ys, color=t["accent"], linewidth=2, zorder=3, solid_capstyle="round")
    # A 2px surface ring keeps markers legible where the line passes behind them.
    ax.plot(
        xs,
        ys,
        "o",
        color=t["accent"],
        markersize=7,
        markeredgecolor=t["surface"],
        markeredgewidth=2,
        zorder=4,
    )

    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("False positives per scan", color=t["muted"], fontsize=9)
    ax.set_ylabel("Sensitivity", color=t["muted"], fontsize=9)
    ax.set_title(
        "Detection: LUNA16 FROC, subset0",
        color=t["ink"],
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=26,
    )

    cpm, lo, hi = froc["cpm"], froc["ci95"][0], froc["ci95"][1]
    ax.annotate(
        f"CPM {cpm:.4f}\n95% CI {lo:.2f} to {hi:.2f}",
        xy=(0.97, 0.06),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        color=t["ink"],
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": t["surface"],
            "edgecolor": t["grid"],
            "linewidth": 1,
        },
    )
    ax.annotate(
        f"{froc['n_scans']} scans, {froc['n_nodules_included']} nodules, official LUNA16 scorer",
        xy=(0, 1.018),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        color=t["muted"],
        fontsize=8.5,
    )


def _panel_metrics(ax, metrics: list[dict], t: dict) -> None:
    ys = list(range(len(metrics)))[::-1]
    for y, m in zip(ys, metrics, strict=True):
        ax.plot(
            [m["low"], m["high"]],
            [y, y],
            color=t["interval"],
            linewidth=2,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            m["value"],
            y,
            "o",
            color=t["accent"],
            markersize=9,
            markeredgecolor=t["surface"],
            markeredgewidth=2,
            zorder=4,
        )
        ax.annotate(
            f"{m['value']:.3f}".rstrip("0").rstrip("."),
            xy=(m["value"], y + 0.15),
            ha="center",
            va="bottom",
            color=t["ink"],
            fontsize=10,
            fontweight="bold",
        )
        ax.annotate(
            m["note"],
            xy=(0.015, y - 0.20),
            ha="left",
            va="top",
            color=t["muted"],
            fontsize=8.5,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([m["label"] for m in metrics], color=t["ink"], fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.65, len(metrics) - 0.35)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Score", color=t["muted"], fontsize=9)
    # y is categorical here, so horizontal gridlines carry no information and would
    # collide with the interval marks. Vertical only.
    ax.grid(False)
    ax.grid(True, axis="x", color=t["grid"], linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_title(
        "Per-stage results, with stated uncertainty",
        color=t["ink"],
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=26,
    )
    ax.annotate(
        "interval type differs per row, see the note under each",
        xy=(0, 1.018),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        color=t["muted"],
        fontsize=8.5,
    )


def _style(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)


def render(theme_name: str, out_dir: Path) -> Path:
    t = THEMES[theme_name]
    froc = json.loads(FROC_JSON.read_text(encoding="utf-8"))

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.2, 4.1), dpi=200, gridspec_kw={"width_ratios": [1.05, 1]}
    )
    fig.patch.set_facecolor(t["surface"])
    for ax in (ax_l, ax_r):
        _style(ax, t)

    _panel_froc(ax_l, froc, t)
    _panel_metrics(ax_r, _headline_metrics(froc), t)

    fig.tight_layout(pad=1.6, w_pad=3.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmarks_{theme_name}.png"
    fig.savefig(out_path, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "img"))
    args = ap.parse_args()

    for name, t in THEMES.items():
        ink = contrast_ratio(t["ink"], t["surface"])
        accent = contrast_ratio(t["accent"], t["surface"])
        muted = contrast_ratio(t["muted"], t["surface"])
        print(
            f"[{name}] contrast vs surface -> ink {ink:.2f}, accent {accent:.2f}, muted {muted:.2f}"
        )
        assert ink >= 4.5, f"{name}: ink fails WCAG AA ({ink:.2f})"
        assert accent >= 3.0, f"{name}: accent fails non-text contrast ({accent:.2f})"
        assert muted >= 4.5, f"{name}: muted text fails WCAG AA ({muted:.2f})"

    for name in THEMES:
        print("wrote", render(name, Path(args.out)))


if __name__ == "__main__":
    main()
