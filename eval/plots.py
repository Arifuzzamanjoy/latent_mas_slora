"""
Comparison plots.

Written to PNG at the end of a run (and, with --live-plot, while it runs).
Every figure answers one question; none of them is decoration:

  accuracy.png     which method is more accurate, and is the gap bigger than
                   the interval
  efficiency.png   what each point of accuracy costs in tokens
  latency.png      p50/p95 wall-clock per item
  by_domain.png    where the difference actually comes from
  deltas.png       paired difference vs the reference, with CI and significance
  calibration.png  does the self-consistency vote share mean anything
  scaling.png      accuracy vs fraction of data (needs --fractions)
  live.png         running accuracy while the eval is still going

Colour follows the data's job, not the series index: magnitude comparisons use
one blue hue with the reference method in emphasis grey, the paired-delta chart
uses a blue/red diverging pair around zero, and only the genuinely multi-series
line charts (scaling, live) use the categorical ramp in fixed slot order.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Categorical ramp, fixed slot order (never cycled, never generated).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

BLUE = "#2a78d6"
BLUE_DARK = "#1c5cab"
RED = "#e34948"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
EMPHASIS_GREY = "#b8b6ae"

_MISSING_MPL = None


def _mpl():
    """Import matplotlib lazily with the non-interactive backend."""
    global _MISSING_MPL
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # pragma: no cover
        _MISSING_MPL = str(e)
        return None


def _style(
    ax, plt, xlabel: str = "", ylabel: str = "", title: str = "", grid_axis: str = "x"
) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9, labelpad=8)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)


def _fig(plt, w: float, h: float):
    fig = plt.figure(figsize=(w, h), dpi=160, facecolor=SURFACE)
    return fig


def _save(fig, path: Path, plt) -> str:
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _scored(summary: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Methods that produce an accuracy, excluding the router classifier."""
    return [(n, m) for n, m in summary["methods"].items() if n != "router-only" and m.get("n")]


def _legend_below(ax, plt, ncol: int = 4) -> None:
    """Legend under the axes so it never covers the data."""
    leg = ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
        handlelength=1.6,
        columnspacing=1.6,
    )
    for t in leg.get_texts():
        t.set_color(INK_2)


def _declutter(fig, texts, pad: float = 2.0, max_iter: int = 200) -> None:
    """
    Push annotations apart until nothing overlaps - measured, not guessed.

    Offsets are nudged in the *rendered* geometry: the text extents are read
    back from the renderer and separated iteratively, so the result holds for
    any label length, font, figure size or data range. Guessing a fixed offset
    in data units (the previous approach) breaks the moment two series converge.
    """
    texts = [t for t in texts if t.get_text()]
    if len(texts) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.get_dpi()
    px_to_pt = 72.0 / dpi

    for _ in range(max_iter):
        boxes = [t.get_window_extent(renderer).expanded(1.0, 1.0) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                if bi.x1 + pad < bj.x0 or bj.x1 + pad < bi.x0:
                    continue  # no horizontal overlap
                overlap = min(bi.y1, bj.y1) - max(bi.y0, bj.y0) + pad
                if overlap <= 0:
                    continue
                shift = (overlap / 2.0) * px_to_pt
                up, down = (i, j) if bi.y0 >= bj.y0 else (j, i)
                for idx, sign in ((up, 1.0), (down, -1.0)):
                    dx, dy = texts[idx].xyann
                    texts[idx].xyann = (dx, dy + sign * shift)
                moved = True
        if not moved:
            return
        fig.canvas.draw()


def _end_labels(ax, entries, max_direct: int = 4):
    """Direct-label line ends. Overlaps are resolved later by _declutter."""
    if len(entries) > max_direct:
        return []
    anns = []
    for x, y, text in sorted(entries, key=lambda e: -e[1]):
        anns.append(
            ax.annotate(
                text,
                (x, y),
                textcoords="offset points",
                xytext=(9, 0),
                color=INK,
                fontsize=9,
                va="center",
                annotation_clip=False,
            )
        )
    return anns


# ─── 1. Accuracy with intervals ──────────────────────────────────────────────


def plot_accuracy(summary: Dict[str, Any], out: Path) -> Optional[str]:
    plt = _mpl()
    if plt is None:
        return None
    rows = _scored(summary)
    if not rows:
        return None
    rows.sort(key=lambda r: r[1]["accuracy"])

    ref = summary.get("reference_method")
    names = [n for n, _ in rows]
    acc = [m["accuracy"] * 100 for _, m in rows]
    lo = [max(0.0, a - m["ci"]["low"] * 100) for a, (_, m) in zip(acc, rows)]
    hi = [max(0.0, m["ci"]["high"] * 100 - a) for a, (_, m) in zip(acc, rows)]

    fig = _fig(plt, 8.2, 0.55 * len(rows) + 2.0)
    ax = fig.add_subplot(111)
    y = range(len(rows))
    # one hue for magnitude; the reference method recedes to grey
    colors = [EMPHASIS_GREY if n == ref else BLUE for n in names]
    ax.barh(list(y), acc, height=0.55, color=colors, zorder=2)
    ax.errorbar(
        acc,
        list(y),
        xerr=[lo, hi],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1.4,
        capsize=4,
        capthick=1.4,
        zorder=3,
    )

    # labels in one right-aligned column rather than trailing each error bar,
    # so they read as a table beside the chart
    reach = max([a + h for a, h in zip(acc, hi)] + [1.0])
    xmax = min(118.0, reach * 1.30 + 6)
    labels = [
        ax.annotate(
            f"{a:.1f}%  (n={m['n']})",
            (xmax, i),
            textcoords="offset points",
            xytext=(0, 0),
            va="center",
            ha="right",
            color=INK,
            fontsize=9,
            annotation_clip=False,
        )
        for i, (a, (_, m)) in enumerate(zip(acc, rows))
    ]

    ax.set_yticks(list(y))
    ax.set_yticklabels([n + ("  ·ref" if n == ref else "") for n in names], color=INK, fontsize=10)
    ax.set_xlim(0, xmax)
    seg = summary["segment"]
    _style(
        ax,
        plt,
        xlabel="accuracy (%)  ·  bars show the 95% Wilson interval",
        title=f"Accuracy — {seg['spec']}, n={seg['n']}",
    )
    _declutter(fig, labels)
    return _save(fig, out / "accuracy.png", plt)


# ─── 2. Cost of a correct answer ─────────────────────────────────────────────


def plot_efficiency(summary: Dict[str, Any], out: Path) -> Optional[str]:
    plt = _mpl()
    if plt is None:
        return None
    rows = _scored(summary)
    if not rows:
        return None

    fig = _fig(plt, 8.2, 5.4)
    ax = fig.add_subplot(111)
    xs = [m["tokens"]["total_mean"] for _, m in rows]
    ys = [m["accuracy"] * 100 for _, m in rows]
    sizes = [max(60, min(520, m["latency_ms"]["p50"] / 12)) for _, m in rows]

    # single hue: identity is carried by the direct label, not by colour, so
    # this stays legible past the three-series cap for all-pairs forms
    ax.scatter(
        xs, ys, s=sizes, color=BLUE, alpha=0.75, linewidths=1.6, edgecolors=SURFACE, zorder=3
    )

    # headroom so labels never run off the axis
    xlo, xhi = min(xs), max(xs)
    pad = max((xhi - xlo) * 0.18, max(1.0, xhi * 0.06))
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(-6, max(105.0, max(ys) + 12))
    xmid = (xlo + xhi) / 2

    anns = []
    for x, y, (n, _) in sorted(zip(xs, ys, rows), key=lambda t: (t[0], t[1])):
        right = x > xmid
        anns.append(
            ax.annotate(
                n,
                (x, y),
                textcoords="offset points",
                xytext=(-10 if right else 10, 7),
                ha="right" if right else "left",
                color=INK,
                fontsize=9,
                annotation_clip=False,
            )
        )
    _declutter(fig, anns)

    _style(
        ax,
        plt,
        xlabel="tokens per item (prompt + completion)",
        ylabel="accuracy (%)",
        grid_axis="both",
        title="Accuracy vs token cost",
    )
    ax.text(
        0.995,
        -0.135,
        "marker area ∝ median latency  ·  up and to the left is better",
        transform=ax.transAxes,
        ha="right",
        color=INK_MUTED,
        fontsize=8,
    )
    return _save(fig, out / "efficiency.png", plt)


# ─── 3. Latency ──────────────────────────────────────────────────────────────


def plot_latency(summary: Dict[str, Any], out: Path) -> Optional[str]:
    plt = _mpl()
    if plt is None:
        return None
    rows = _scored(summary)
    if not rows:
        return None
    rows.sort(key=lambda r: r[1]["latency_ms"]["p50"])

    fig = _fig(plt, 8.2, 0.55 * len(rows) + 2.0)
    ax = fig.add_subplot(111)
    y = list(range(len(rows)))
    p50 = [m["latency_ms"]["p50"] / 1000 for _, m in rows]
    p95 = [m["latency_ms"]["p95"] / 1000 for _, m in rows]

    # two shades of one hue: same measure, two quantiles - not two scales
    ax.barh([i + 0.16 for i in y], p95, height=0.3, color="#9ec5f4", zorder=2, label="p95")
    ax.barh([i - 0.16 for i in y], p50, height=0.3, color=BLUE_DARK, zorder=2, label="p50")
    xmax = max(p95 + [0.1]) * 1.32
    labels = [
        ax.annotate(
            f"{a:.1f}s / {b:.1f}s",
            (xmax, i),
            textcoords="offset points",
            xytext=(0, 0),
            va="center",
            ha="right",
            color=INK,
            fontsize=9,
            annotation_clip=False,
        )
        for i, (a, b) in enumerate(zip(p50, p95))
    ]

    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in rows], color=INK, fontsize=10)
    ax.set_xlim(0, xmax)
    leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    _style(ax, plt, xlabel="seconds per item", title="Latency per item")
    _declutter(fig, labels)
    return _save(fig, out / "latency.png", plt)


# ─── 4. Per-domain small multiples ───────────────────────────────────────────


def plot_by_domain(summary: Dict[str, Any], out: Path) -> Optional[str]:
    plt = _mpl()
    if plt is None:
        return None
    rows = _scored(summary)
    domains = sorted(summary["segment"].get("by_domain", {}))
    if not rows or len(domains) < 2:
        return None

    # one panel per domain rather than one colour per method: keeps a single
    # hue and stays readable well past any categorical cap
    fig = _fig(plt, 4.1 * len(domains), 0.42 * len(rows) + 2.2)
    order = sorted(rows, key=lambda r: r[1]["accuracy"], reverse=True)
    names = [n for n, _ in order]

    for k, dom in enumerate(domains):
        ax = fig.add_subplot(1, len(domains), k + 1)
        vals, ns = [], []
        for _name, m in order:
            d = m.get("by_domain", {}).get(dom)
            vals.append(d["accuracy"] * 100 if d else 0.0)
            ns.append(d["n"] if d else 0)
        y = list(range(len(names)))[::-1]
        ax.barh(y, vals, height=0.55, color=BLUE, zorder=2)
        panel_labels = [
            ax.annotate(
                f"{v:.0f}%",
                (v + 2, yy),
                textcoords="offset points",
                xytext=(0, 0),
                va="center",
                color=INK,
                fontsize=8,
                annotation_clip=False,
            )
            for yy, v in zip(y, vals)
        ]
        _declutter(fig, panel_labels)
        ax.set_yticks(y)
        ax.set_yticklabels(names if k == 0 else [""] * len(names), color=INK, fontsize=9)
        ax.set_xlim(0, 112)
        _style(
            ax,
            plt,
            xlabel="accuracy (%)",
            title=f"{dom}  (n={summary['segment']['by_domain'].get(dom, max(ns) if ns else 0)})",
        )

    fig.suptitle("Accuracy by domain", color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out / "by_domain.png", plt)


# ─── 5. Paired deltas ────────────────────────────────────────────────────────


def plot_deltas(summary: Dict[str, Any], out: Path) -> Optional[str]:
    plt = _mpl()
    if plt is None:
        return None
    comps = summary.get("comparisons") or {}
    if not comps:
        return None
    ref = summary.get("reference_method")
    items = sorted(comps.items(), key=lambda kv: kv[1]["delta"])

    fig = _fig(plt, 8.2, 0.55 * len(items) + 2.2)
    ax = fig.add_subplot(111)
    y = list(range(len(items)))
    deltas = [c["delta"] * 100 for _, c in items]
    lo = [max(0.0, d - c["ci_low"] * 100) for d, (_, c) in zip(deltas, items)]
    hi = [max(0.0, c["ci_high"] * 100 - d) for d, (_, c) in zip(deltas, items)]

    # diverging around zero: polarity is the whole point of this chart
    colors = [BLUE if d >= 0 else RED for d in deltas]
    ax.barh(y, deltas, height=0.55, color=colors, zorder=2)
    ax.errorbar(
        deltas,
        y,
        xerr=[lo, hi],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1.4,
        capsize=4,
        capthick=1.4,
        zorder=3,
    )
    ax.axvline(0, color=INK_2, linewidth=1.2, zorder=4)

    span = max([abs(d) + h for d, h in zip(deltas, hi)] + [1.0])
    labels = []
    for i, (_name, c) in enumerate(items):
        p = c["mcnemar"]["p_value"]
        star = "  p<0.05" if p < 0.05 else f"  p={p:.2f}"
        side = 1 if deltas[i] >= 0 else -1
        labels.append(
            ax.annotate(
                f"{deltas[i]:+.1f}pp{star}",
                (deltas[i] + side * (hi[i] + span * 0.05), i),
                textcoords="offset points",
                xytext=(0, 0),
                va="center",
                ha="left" if side > 0 else "right",
                color=INK,
                fontsize=9,
                annotation_clip=False,
            )
        )

    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in items], color=INK, fontsize=10)
    ax.set_xlim(-span * 1.45, span * 1.45)
    _style(
        ax,
        plt,
        xlabel="accuracy difference (percentage points), paired bootstrap CI",
        title=f"Difference vs {ref} — same items, McNemar test",
    )
    _declutter(fig, labels)
    return _save(fig, out / "deltas.png", plt)


# ─── 6. Reliability diagram ──────────────────────────────────────────────────


def plot_calibration(summary: Dict[str, Any], out: Path) -> Optional[str]:
    plt = _mpl()
    if plt is None:
        return None
    rows = [(n, m) for n, m in _scored(summary) if m.get("calibration", {}).get("bins")]
    rows = [(n, m) for n, m in rows if len(m["calibration"]["bins"]) > 1]
    if not rows:
        return None  # single-sample runs have no spread

    cols = min(3, len(rows))
    graph_rows = (len(rows) + cols - 1) // cols
    fig = _fig(plt, 3.5 * cols, 3.4 * graph_rows)
    for i, (name, m) in enumerate(rows):
        ax = fig.add_subplot(graph_rows, cols, i + 1)
        bins = m["calibration"]["bins"]
        xs = [b["avg_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.plot(
            xs,
            ys,
            color=BLUE,
            linewidth=2.0,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        _style(
            ax,
            plt,
            xlabel="vote share",
            ylabel="accuracy" if i % cols == 0 else "",
            grid_axis="both",
            title=f"{name}  ECE {m['calibration']['ece']}",
        )

    fig.suptitle(
        "Reliability — does agreement predict correctness?",
        color=INK,
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out / "calibration.png", plt)


# ─── 7. Data-scaling curve ───────────────────────────────────────────────────


def plot_scaling(
    points: Dict[str, List[Tuple[float, float, int]]], out: Path, dataset: str = ""
) -> Optional[str]:
    """points: {method: [(fraction, accuracy, n), ...]}"""
    plt = _mpl()
    if plt is None or not points:
        return None

    fig = _fig(plt, 8.4, 5.4)
    ax = fig.add_subplot(111)
    ends: List[Tuple[float, float, str]] = []
    for i, (name, pts) in enumerate(sorted(points.items())):
        pts = sorted(pts)
        xs = [p[0] * 100 for p in pts]
        ys = [p[1] * 100 for p in pts]
        color = SERIES[i % len(SERIES)]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
            label=name,
        )
        ends.append((xs[-1], ys[-1], name))

    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    _declutter(fig, _end_labels(ax, ends))
    _legend_below(ax, plt, ncol=min(4, len(points)))
    _style(
        ax,
        plt,
        xlabel="fraction of dataset evaluated (%, log scale)",
        ylabel="accuracy (%)",
        grid_axis="both",
        title=f"Accuracy vs amount of data{(' — ' + dataset) if dataset else ''}",
    )
    ax.text(
        0.995,
        -0.30,
        "slices are nested: each larger fraction adds items, never swaps them",
        transform=ax.transAxes,
        ha="right",
        color=INK_MUTED,
        fontsize=8,
    )
    return _save(fig, out / "scaling.png", plt)


# ─── 8. Live progress ────────────────────────────────────────────────────────


def plot_live(
    records: List[Dict[str, Any]],
    out: Path,
    total: Optional[int] = None,
    title: str = "Running accuracy",
) -> Optional[str]:
    """Running accuracy per method, refreshed while the eval is in flight."""
    plt = _mpl()
    if plt is None or not records:
        return None

    series: Dict[str, List[float]] = {}
    for r in records:
        s = series.setdefault(r["method"], [])
        s.append(1.0 if r["correct"] else 0.0)

    from matplotlib.ticker import MaxNLocator

    fig = _fig(plt, 8.4, 5.0)
    ax = fig.add_subplot(111)
    ends: List[Tuple[float, float, str]] = []
    longest = 0
    for i, (name, vals) in enumerate(sorted(series.items())):
        run, acc = 0.0, []
        for k, v in enumerate(vals, start=1):
            run += v
            acc.append(100 * run / k)
        longest = max(longest, len(acc))
        color = SERIES[i % len(SERIES)]
        ax.plot(range(1, len(acc) + 1), acc, color=color, linewidth=2.0, zorder=3, label=name)
        ends.append((len(acc), acc[-1], f"{name} {acc[-1]:.0f}%"))

    ax.set_ylim(0, 105)
    ax.set_xlim(1, max(total or longest, 2))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _declutter(fig, _end_labels(ax, ends))
    _legend_below(ax, plt, ncol=min(4, len(series)))
    # Methods run sequentially, so records/methods under-reports the one in
    # flight and over-reports the ones not started. Show each method's own
    # progress instead.
    done = len(records)
    parts = [f"{n} {len(v)}" + (f"/{total}" if total else "") for n, v in sorted(series.items())]
    subtitle = "  ·  ".join(parts[:4]) + ("  ·  …" if len(parts) > 4 else "")
    _style(
        ax,
        plt,
        xlabel="items scored per method",
        ylabel="running accuracy (%)",
        grid_axis="both",
        title=f"{title} — {done} records",
    )
    # re-set the title with extra pad so the per-method line fits beneath it
    ax.set_title(
        f"{title} — {done} records", color=INK, fontsize=12, fontweight="bold", loc="left", pad=30
    )
    ax.text(
        0, 1.012, subtitle, transform=ax.transAxes, ha="left", va="bottom", color=INK_2, fontsize=9
    )
    return _save(fig, out / "live.png", plt)


# ─── Entry point ─────────────────────────────────────────────────────────────


def make_all(summary: Dict[str, Any], out_dir: Path) -> List[str]:
    """Write every applicable figure. Never raises - plots are not the result."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    for fn in (
        plot_accuracy,
        plot_efficiency,
        plot_latency,
        plot_by_domain,
        plot_deltas,
        plot_calibration,
    ):
        try:
            p = fn(summary, out)
            if p:
                made.append(p)
        except Exception as e:  # pragma: no cover
            print(f"[plots] {fn.__name__} failed: {type(e).__name__}: {e}")
    if _MISSING_MPL:
        print(f"[plots] matplotlib unavailable ({_MISSING_MPL}); skipped figures")
    return made


def scaling_from_runs(run_dirs: Sequence[Path], out_dir: Path) -> Optional[str]:
    """Build the scaling curve from several completed run directories."""
    points: Dict[str, List[Tuple[float, float, int]]] = {}
    dataset = ""
    for d in run_dirs:
        f = Path(d) / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        frac = float(s["config"].get("fraction", 1.0))
        dataset = s["segment"].get("spec", dataset)
        for name, m in s["methods"].items():
            if name == "router-only" or not m.get("n"):
                continue
            points.setdefault(name, []).append((frac, m["accuracy"], m["n"]))
    if not points:
        return None
    return plot_scaling(points, Path(out_dir), dataset)
