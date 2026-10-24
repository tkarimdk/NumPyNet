"""Portable CSV experiment logs and dependency-free SVG loss curves."""

from __future__ import annotations

import csv
from html import escape
from math import log10
from pathlib import Path

from .experiments import EpochRecord


def write_history_csv(path: str | Path, history: list[EpochRecord]) -> Path:
    """Write epoch-zero and post-update metrics in a machine-readable format."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "loss", "accuracy"])
        for row in history:
            writer.writerow([row.epoch, format(row.loss, ".12g"), format(row.accuracy, ".12g")])
    return destination


def write_comparison_csv(path: str | Path, results: list[tuple[float, list[EpochRecord]]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["learning_rate", "epochs", "initial_loss", "final_loss", "final_accuracy"])
        for rate, rows in results:
            writer.writerow([rate, rows[-1].epoch, f"{rows[0].loss:.12g}",
                             f"{rows[-1].loss:.12g}", f"{rows[-1].accuracy:.12g}"])
    return destination


def write_loss_svg(path: str | Path, results: list[tuple[float | str, list[EpochRecord]]]) -> Path:
    """Write a browser-openable logarithmic loss plot, without matplotlib.

    Each curve uses the same network seed if produced by compare-lr.
    """
    if not results or any(not rows for _, rows in results):
        raise ValueError("A plot needs one or more nonempty histories")
    if any(not all(row.loss > 0 for row in rows) for _, rows in results):
        raise ValueError("Loss must be positive for a logarithmic plot")
    if len(results) > 8:
        raise ValueError("At most eight curves are supported by the chart palette")

    import math

    if any(not math.isfinite(row.loss) for _, rows in results for row in rows):
        raise ValueError("Loss values must be finite")

    width, height = 940, 540
    left, right, top, bottom = 88, 28, 64, 84
    chart_w, chart_h = width - left - right, height - top - bottom
    max_epoch = max(rows[-1].epoch for _, rows in results)
    if max_epoch <= 0:
        raise ValueError("At least one training epoch is required")
    logs = [log10(row.loss) for _, rows in results for row in rows]
    y_min, y_max = min(logs), max(logs)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    padding = (y_max - y_min) * 0.07
    y_min -= padding
    y_max += padding

    def xx(epoch: int) -> float:
        return left + chart_w * epoch / max_epoch

    def yy(loss: float) -> float:
        return top + chart_h * (y_max - log10(loss)) / (y_max - y_min)

    palette = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0d9488", "#dc2626", "#4f46e5"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="88" y="36" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#0f172a">XOR learning-rate experiment</text>',
        '<text x="88" y="53" font-family="Arial,sans-serif" font-size="12" fill="#64748b">Training loss on 4 XOR samples · log10 scale · not a generalization metric</text>',
    ]
    for i in range(5):
        level = y_min + (y_max - y_min) * i / 4
        py = top + chart_h * (1 - i / 4)
        svg.append(f'<line x1="{left}" x2="{left+chart_w}" y1="{py:.2f}" y2="{py:.2f}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{left-12}" y="{py+4:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#475569">{10**level:.2g}</text>')
    for i in range(5):
        epoch = round(i * max_epoch / 4)
        px = xx(epoch)
        svg.append(f'<text x="{px:.2f}" y="{top+chart_h+23}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#475569">{epoch}</text>')
    svg.append(f'<text x="{left+chart_w/2}" y="{height-27}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#334155">Training epoch</text>')
    svg.append(f'<text transform="translate(23 {top+chart_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#334155">Binary cross-entropy loss (log scale)</text>')
    svg.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+chart_h}" stroke="#64748b"/>')
    svg.append(f'<line x1="{left}" x2="{left+chart_w}" y1="{top+chart_h}" y2="{top+chart_h}" stroke="#64748b"/>')
    for i, (rate, rows) in enumerate(results):
        # For large experiments, take evenly spaced points and always retain the last one.
        stride = max(1, (len(rows) - 1 + 699) // 700)
        plotted = rows[::stride]
        if plotted[-1] is not rows[-1]:
            plotted = [*plotted, rows[-1]]
        points = " ".join(f"{xx(row.epoch):.2f},{yy(row.loss):.2f}" for row in plotted)
        color = palette[i]
        svg.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>')
        legend_x = 500 + (i % 3) * 144
        legend_y = 19 + (i // 3) * 17
        label = escape(rate if isinstance(rate, str) else f"η={rate:g}")
        svg.append(f'<line x1="{legend_x}" x2="{legend_x+20}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x+25}" y="{legend_y+4}" font-family="Arial,sans-serif" font-size="12" fill="#334155">{label}</text>')
    svg.append('</svg>')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return destination
