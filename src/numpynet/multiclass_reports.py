"""Portable results for the multiclass demonstration."""

from __future__ import annotations

import csv
from pathlib import Path

from .multiclass import MulticlassRecord


def write_multiclass_csv(path: str | Path, history: list[MulticlassRecord]) -> Path:
    if not history:
        raise ValueError("History cannot be empty")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"])
        for item in history:
            writer.writerow([item.epoch, f"{item.train_loss:.12g}", f"{item.train_accuracy:.12g}",
                             f"{item.validation_loss:.12g}", f"{item.validation_accuracy:.12g}"])
    return target


def write_multiclass_svg(path: str | Path, history: list[MulticlassRecord]) -> Path:
    """Generate an offline chart with clearly separated training/validation losses."""
    import math

    if len(history) < 2 or history[-1].epoch <= 0:
        raise ValueError("Need at least one update to draw a loss chart")
    losses = [value for record in history for value in (record.train_loss, record.validation_loss)]
    if not all(math.isfinite(value) and value >= 0 for value in losses):
        raise ValueError("Losses must be finite and nonnegative")

    width, height = 900, 500
    left, right, top, bottom = 84, 35, 80, 72
    chart_width = width - left - right
    chart_height = height - top - bottom
    top_loss = max(losses) * 1.08 or 1.0
    last_epoch = history[-1].epoch

    def x(epoch: int) -> float:
        return left + chart_width * epoch / last_epoch

    def y(loss: float) -> float:
        return top + chart_height * (1 - loss / top_loss)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="84" y="34" font-family="Arial,sans-serif" font-weight="700" font-size="22" fill="#172033">Synthetic three-class spiral: training vs validation</text>',
        '<text x="84" y="56" font-family="Arial,sans-serif" font-size="12" fill="#4b5563">Independent generated holdout, same synthetic distribution; not a real-world benchmark</text>',
    ]
    for tick in range(5):
        value = top_loss * tick / 4
        yy = y(value)
        lines.extend([
            f'<line x1="{left}" x2="{width-right}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#e2e8f0"/>',
            f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" font-size="12" font-family="Arial,sans-serif" fill="#475569">{value:.3g}</text>',
        ])
        xx = x(round(last_epoch * tick / 4))
        lines.append(f'<text x="{xx:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#475569">{round(last_epoch * tick/4)}</text>')
    for i, (name, field, color) in enumerate([
        ("Training", "train_loss", "#2563eb"),
        ("Validation", "validation_loss", "#d97706"),
    ]):
        stride = max(1, (len(history) + 699) // 700)
        sampled = history[::stride]
        if sampled[-1] is not history[-1]:
            sampled = [*sampled, history[-1]]
        pts = " ".join(f"{x(record.epoch):.2f},{y(getattr(record, field)):.2f}" for record in sampled)
        lines.append(f'<polyline points="{pts}" stroke="{color}" stroke-width="2.6" fill="none" stroke-linejoin="round"/>')
        lines.append(f'<line x1="{left+25+i*150}" x2="{left+45+i*150}" y1="{height-25}" y2="{height-25}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{left+53+i*150}" y="{height-21}" font-family="Arial,sans-serif" font-size="12" fill="#334155">{name}</text>')
    lines.extend([
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#64748b"/>',
        f'<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#64748b"/>',
        f'<text x="{left+chart_width/2:.1f}" y="{height-43}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#334155">Full-batch parameter updates</text>',
        '</svg>',
    ])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
