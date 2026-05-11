"""
Executive deck visualization defaults for notebooks 02–08.
Import after setting PROJECT_ROOT / os.chdir to repo root.
"""
from __future__ import annotations

import datetime as _dt

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

PALETTE = {
    "uber": "#000000",
    "lyft": "#FF00BF",
    "crz": "#06C167",
    "non_crz": "#CCCCCC",
    "accent": "#1F77B4",
    "warn": "#D62728",
    "muted": "#999999",
    "annot_bg": "#FFFFE0",
}


def set_rcparams() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def add_footnote(
    fig,
    text: str,
    y: float = -0.02,
    fontsize: int = 9,
    color: str = "#888888",
    style: str = "italic",
) -> None:
    fig.text(
        0.5, y, text, ha="center", va="top", fontsize=fontsize,
        color=color, style=style, wrap=False,
    )


def annotate_callout(
    ax,
    xy,
    text: str,
    xytext: tuple[float, float] = (12, 12),
    bbox=None,
    fontsize: int = 9,
):
    if bbox is None:
        bbox = dict(boxstyle="round", fc=PALETTE["annot_bg"], ec="#CCCCCC", alpha=0.95)
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="offset points",
        bbox=bbox,
        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=1.1),
        fontsize=fontsize,
    )


def format_thousands_axis(ax, axis: str = "y") -> None:
    fmt = ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)
    getattr(ax, f"{axis}axis").set_major_locator(ticker.MaxNLocator(nbins=7))


def format_pct_axis(ax, axis: str = "y", decimals: int = 1) -> None:
    fmt = ticker.FuncFormatter(lambda x, _: f"{x:.{decimals}f}%")
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)
    getattr(ax, f"{axis}axis").set_major_locator(ticker.MaxNLocator(nbins=7))


def format_coef_axis(ax, axis: str = "x", decimals: int = 4) -> None:
    """Regression coefficients (not scaled to %)."""
    fmt = ticker.FuncFormatter(lambda x, _: f"{x:.{decimals}f}")
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)
    getattr(ax, f"{axis}axis").set_major_locator(ticker.MaxNLocator(nbins=7))


def format_dollar_axis(ax, axis: str = "y") -> None:
    fmt = ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)
    getattr(ax, f"{axis}axis").set_major_locator(ticker.MaxNLocator(nbins=7))


def direct_label_lines(ax, lines, labels: list[str], colors: list | None = None, fontsize: int = 9):
    """Right-edge labels for Line2D objects (numeric or datetime x)."""
    cols = colors or [ln.get_color() for ln in lines]
    for ln, lab, col in zip(lines, labels, cols):
        xd, yd = ln.get_xdata(), ln.get_ydata()
        if len(xd) == 0:
            continue

        v_last = xd[-1]
        is_datetime = isinstance(v_last, (_dt.datetime, _dt.date, np.datetime64))
        if not is_datetime and hasattr(v_last, "to_pydatetime"):
            # pandas.Timestamp and similar
            is_datetime = True

        if is_datetime:
            xd_num = mdates.date2num(np.asarray(xd))
            span_num = (
                float(np.nanmax(xd_num) - np.nanmin(xd_num)) if len(xd_num) > 1 else 1.0
            )
            x_label_pos = mdates.num2date(xd_num[-1] + 0.02 * span_num)
        else:
            span = float(np.nanmax(xd) - np.nanmin(xd)) if len(xd) > 1 else 1.0
            x_label_pos = xd[-1] + 0.02 * span

        ax.text(
            x_label_pos,
            yd[-1],
            lab,
            color=col or PALETTE["uber"],
            fontsize=fontsize,
            va="center",
            ha="left",
        )


def limit_ticks(ax, axis: str = "both", nbins: int = 7) -> None:
    if axis in ("x", "both"):
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
