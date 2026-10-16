# Воспроизводимость

## Проверить опубликованные результаты

Для анализа достаточно `experiments/baseline_rerun/per_image.csv` и `summary.json`.
Они содержат результат каждого кадра, параметры и SHA-256 данных/весов.
Графики baseline и validation строятся командой `python scripts/plot_results.py`
после установки `.[analysis]`. Training loss и сравнение из отчёта:
`python scripts/plot_research.py`. Второй скрипт использует исходный training log
и `experiments/report_findings.json`, где сохранены статус и источник каждого числа.
Исходный код исследования сохранён в `notebooks/metro_experiments.ipynb`;
выводы с большими изображениями удалены из его Git-версии, training log сохранён отдельно.

## Приложение и baseline

Python 3.10–3.12. Установить `.[app]` и запустить `python -m streamlit run app.py`.
Команда `bash scripts/setup_yolov5.sh` устанавливает профиль YOLOv5 v7.0 / PyTorch 2.5.1
и скачивает официальные pretrained веса. Это проверенный профиль приложения.

Чтобы подключить данные:

```bash
python scripts/import_dataset.py dataset.zip --output data/metro
python -m metroyolo.cli audit data/metro
python -m metroyolo.cli evaluate \
  --images data/metro/valid/images --labels data/metro/valid/labels \
  --weights weights/yolov5s.pt --output outputs/baseline
```

Файл `data.yaml` создаётся с абсолютным локальным корнем и классом person.
Состав выборок и байты файлов не меняются. Данные, веса и результаты новых запусков
исключены из Git.

## Обучение и сравнение

После установки `.[inference,training]` доступна команда:

```bash
metroyolo-train --dataset data/metro --device 0 \
  --also-10-epochs --full-sweep --output runs/experiment-01
```

Основной запуск: 30 эпох, batch 16, 640 px, SGD, seed 0. Каждое дообучение начинается
с pretrained весов, создаёт отдельные best.pt/last.pt, логи и CSV по validation.
Полная сетка содержит 350 вариантов. Каталог существующего запуска не перезаписывается.
Для технической проверки на CPU допустимы `--device cpu --workers 0`, уменьшенный
размер и одна эпоха; такие результаты нельзя представлять как основной эксперимент.

## Исходная программная версия

Исторический запуск использовал commit `70b964b6d5067fff621f724c85d0e39e6b4c8e4e`,
PyTorch 2.10.0+cu128 и Ultralytics 8.4.47. Это отличается от профиля приложения v7.0.
При научном повторе нужно отдельное окружение с этими версиями, checkout указанного
commit и аргумент `--repo` для его каталога. Параметры записаны в
`configs/research_protocol.json`, подробности — в исходном логе.

Исторические веса best.pt не сохранились. Повторная тренировка создаёт новый артефакт,
а не восстанавливает прежние тензоры. Все новые метрики измеряются после запуска;
исторические числа не подставляются в выходные CSV.
