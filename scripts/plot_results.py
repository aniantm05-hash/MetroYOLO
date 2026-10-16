"""Generate the portfolio figure from the committed per-image baseline predictions."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "experiments/baseline_rerun/per_image.csv").open() as handle:
        rows = sorted(
            list(csv.DictReader(handle)), key=lambda r: (int(r["true_count"]), r["image"])
        )
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 9))
    y = np.arange(len(rows))
    truth = np.array([int(r["true_count"]) for r in rows])
    pred = np.array([int(r["pred_count"]) for r in rows])
    colors = [
        "#b6463a" if p < t else "#287bb5" if p > t else "#18806b"
        for t, p in zip(truth, pred, strict=True)
    ]
    ax.hlines(y, truth, pred, color=colors, linewidth=3, zorder=2)
    ax.scatter(
        truth,
        y,
        s=100,
        facecolors="white",
        edgecolors="#34454f",
        linewidth=1.5,
        label="Разметка",
        zorder=3,
    )
    ax.scatter(pred, y, s=35, c=colors, label="Предсказание YOLOv5s", zorder=4)
    for i, (t, p) in enumerate(zip(truth, pred, strict=True)):
        ax.text(
            max(t, p) + 0.4,
            i,
            "точно" if t == p else f"{p - t:+d}",
            va="center",
            fontsize=10,
            color=colors[i],
        )
    ax.set_yticks(y, [r["image"].split("_jpg")[0] for r in rows])
    ax.set_xticks(range(0, 19, 2))
    ax.set(xlim=(-0.5, 19), xlabel="Число людей", ylabel="Кадр · по возрастанию числа людей")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e4e9ed")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True)
    fig.suptitle("Кого недосчитала модель?", fontsize=23, x=0.12, ha="left", y=0.97)
    fig.text(
        0.12,
        0.92,
        "Все 21 validation-кадр. Красный — недосчёт, "
        "синий — завышение, зелёный — точный ответ.",
        fontsize=10,
    )
    fig.text(
        0.12,
        0.025,
        "Источник: experiments/baseline_rerun/per_image.csv · без отбора удачных кадров",
        fontsize=10,
        color="#536572",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.87, bottom=0.11)
    target = root / "docs/assets/baseline-analysis"
    fig.savefig(target.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(target.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def training_history():
    root = Path(__file__).resolve().parents[1]
    log = (root / "experiments/notebook_training_log.txt").read_text()
    pattern = r"^\s*all\s+21\s+159\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    measurements = re.findall(pattern, log, re.MULTILINE)
    if len(measurements) != 31:
        raise ValueError("Expected 30 epoch validation rows and one final checkpoint row")
    # The last row validates best.pt after training; it is not a 31st epoch.
    rows = [
        {
            "epoch": i,
            **dict(
                zip(["precision", "recall", "map50", "map50_95"], map(float, values), strict=True)
            ),
        }
        for i, values in enumerate(measurements[:30], 1)
    ]
    with (root / "experiments/training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Детектор учится. Стал ли он лучше считать?",
        x=0.065,
        ha="left",
        fontsize=20,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.065,
        0.86,
        "30 эпох · исходный лог · validation после каждой эпохи · без сглаживания",
        color="#536572",
        fontsize=12,
    )
    series = [
        ("precision", "Precision", "#18806b"),
        ("recall", "Recall", "#287bb5"),
        ("map50", "mAP@0.5", "#18806b"),
        ("map50_95", "mAP@0.5:0.95", "#287bb5"),
    ]
    for index, (key, label, color) in enumerate(series):
        ax = axes[index // 2]
        ax.plot(
            [r["epoch"] for r in rows],
            [r[key] for r in rows],
            label=label,
            color=color,
            linewidth=2,
        )
    for ax, title in zip(
        axes, ["Точность и полнота детекции", "Качество локализации"], strict=True
    ):
        ax.set(xlim=(1, 30), ylim=(0, 1), xlabel="Эпоха", ylabel="Значение метрики")
        ax.set_title(title, loc="left", pad=15, color="#233642")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dbe2e8", alpha=0.6)
        ax.legend(frameon=False, labelcolor="#536572")
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.24, top=0.72, wspace=0.3)
    fig.text(
        0.065,
        0.035,
        "Эти кривые измеряют детекцию, а не ошибку подсчёта. "
        "Сравнение с baseline требует отдельной оценки.",
        fontsize=10,
        color="#536572",
    )
    target = root / "docs/assets/training-history"
    fig.savefig(target.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(target.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
    training_history()
