from __future__ import annotations

import argparse
import json
from textwrap import fill
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Не установлены зависимости. "
        "Выполните: pip install pandas matplotlib seaborn"
    ) from exc


def load_events(json_path: Path) -> pd.DataFrame:
    """Загружает события из JSON и проверяет базовую целостность данных."""
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("В JSON ожидается ключ 'events' со списком событий.")

    df = pd.DataFrame(events)
    if df.empty:
        raise ValueError("Список событий пуст. Нечего анализировать.")

    required_columns = {"timestamp", "signature"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"В данных отсутствуют обязательные поля: {missing_list}"
        )

    # Ошибки парсинга преобразуются в NaT, чтобы их можно было явно отловить.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Найдены некорректные значения в поле 'timestamp'.")

    df["signature"] = df["signature"].astype(str).str.strip()
    if (df["signature"] == "").any():
        raise ValueError("Найдены пустые значения в поле 'signature'.")

    return df


def build_signature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Считает частоту событий по сигнатурам и долю в процентах."""
    counts = df["signature"].value_counts().sort_values(ascending=False)
    stats = counts.rename("count").to_frame()
    stats["share_%"] = (stats["count"] / len(df) * 100).round(2)
    return stats


def plot_signature_distribution(
    stats: pd.DataFrame, output_path: Path, show: bool = False
) -> None:
    """Строит и сохраняет столбчатую диаграмму распределения сигнатур."""
    sns.set_theme(style="whitegrid")
    plot_data = stats.reset_index()
    plot_data.columns = ["signature", "count", "share_%"]
    plot_data["signature_wrapped"] = plot_data["signature"].apply(
        lambda value: fill(value, width=46)
    )

    fig_height = max(7, len(plot_data) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    colors = sns.color_palette("mako", n_colors=len(plot_data))
    bars = ax.barh(
        plot_data["signature_wrapped"],
        plot_data["count"],
        color=colors,
    )
    ax.invert_yaxis()

    max_count = float(plot_data["count"].max())
    ax.set_xlim(0, max_count * 1.25)
    ax.set_title(
        "Распределение событий ИБ по типам сигнатур",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Количество событий", fontsize=12)
    ax.set_ylabel("Тип события (signature)", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    sns.despine(left=True, bottom=True)

    # Подписи справа от каждого бара показывают count и долю в %.
    for bar, share in zip(bars, plot_data["share_%"]):
        value = int(bar.get_width())
        ax.text(
            bar.get_width() + max_count * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({share:.1f}%)",
            ha="left",
            va="center",
            fontsize=10,
            color="#1F2937",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Возвращает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Анализ и визуализация событий ИБ из events.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("events.json"),
        help="Путь к входному JSON-файлу",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("signature_distribution.png"),
        help="Путь для сохранения графика",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показать график в окне после сохранения",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа: анализ и сохранение графика."""
    args = parse_args()

    df = load_events(args.input)
    stats = build_signature_stats(df)

    plot_signature_distribution(stats, args.plot_output, show=args.show)

    print(f"Событий загружено: {len(df)}")
    print(f"Уникальных сигнатур: {stats.shape[0]}")
    print("\nРаспределение по signature:")
    print(stats.to_string())
    print(f"\nГрафик сохранен: {args.plot_output.resolve()}")


if __name__ == "__main__":
    main()
