"""Plot reported comparison and observed training losses with explicit provenance."""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"docs/assets/{name}.{ext}", dpi=170, facecolor="white")
    plt.close(fig)


def reported_comparison():
    findings = json.loads((ROOT / "experiments/report_findings.json").read_text())
    values = [findings["baseline"]["count_mae"], findings["fine_tuned_reported_run"]["count_mae"]]
    fig, ax = plt.subplots(figsize=(11, 4.7))
    bars = ax.barh([1, 0], values, color=["#18806b", "#b6463a"], height=0.42)
    ax.set_yticks([1, 0], ["Pretrained YOLOv5s", "Дообучение · запуск из отчёта"])
    ax.set(xlim=(0, 2.35), xlabel="Средняя абсолютная ошибка подсчёта, человек · меньше лучше")
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            val + 0.045,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}".replace(".", ","),
            va="center",
            fontsize=19,
            fontweight="bold",
        )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e4e9ed")
    ax.set_axisbelow(True)
    fig.suptitle(
        "Дообучение не улучшило прикладной результат", x=0.035, ha="left", fontsize=21, y=0.97
    )
    fig.text(
        0.035,
        0.83,
        "Сравнение, зафиксированное в исходной практике. Ошибка выросла примерно на 81%.",
        fontsize=11,
    )
    fig.text(
        0.035,
        0.035,
        "Источники: отчёт, стр. 26 и 31. Baseline повторно проверен; "
        "для 1,90 сохранился агрегат, но не checkpoint.",
        fontsize=9,
        color="#536572",
    )
    fig.subplots_adjust(left=0.31, right=0.96, top=0.73, bottom=0.23)
    save(fig, "reported-comparison")


def training_losses():
    log = (ROOT / "experiments/notebook_training_log.txt").read_text()
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    pattern = (
        r"^\s*(\d+)/29\s+([\d.]+G)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        r"\s+\d+\s+640:.*100%.*6/6"
    )
    matches = re.findall(pattern, log, re.MULTILINE)
    if len(matches) != 30 or [int(m[0]) for m in matches] != list(range(30)):
        raise ValueError("Expected the final progress record of each original training epoch")
    rows = [
        {
            "epoch": int(m[0]) + 1,
            "train_box_loss": float(m[2]),
            "train_objectness_loss": float(m[3]),
            "train_class_loss": float(m[4]),
        }
        for m in matches
    ]
    with (ROOT / "experiments/training_losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, label, color in zip(
        axes,
        ["train_box_loss", "train_objectness_loss"],
        ["Ошибка положения рамок · box loss", "Ошибка наличия объекта · objectness loss"],
        ["#18806b", "#287bb5"],
        strict=True,
    ):
        ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], color=color, lw=2)
        ax.set(xlim=(1, 30), ylim=(0, None), xlabel="Эпоха", ylabel="Training loss")
        ax.set_title(label, loc="left", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e4e9ed")
    fig.suptitle("Что оптимизировалось во время обучения", fontsize=21, x=0.07, ha="left", y=0.97)
    fig.text(
        0.07,
        0.83,
        "Исходный 30-эпоховый запуск · значения в конце каждой эпохи · без сглаживания",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.035,
        "Источник: сохранённый training log. Class loss = 0 в одноклассовом запуске; "
        "это не показатель точности.",
        fontsize=9,
        color="#536572",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.22, wspace=0.30)
    save(fig, "training-loss")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    reported_comparison()
    training_losses()
