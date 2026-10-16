# Запуск проекта на Mac

Из папки проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[app]'
python -m streamlit run app.py
```

Откроется исследование: исходное сравнение, ход обучения, короткий запуск,
поиск порогов и анализ ошибок. Во второй вкладке доступны три примера из отчёта.
Для собственной фотографии установи модель командой `bash scripts/setup_yolov5.sh`,
перейди во вкладку «Детекция на фото», выбери режим YOLOv5, CPU и загрузи изображение.

Основной код находится в `src/metroyolo`, исходное исследование — в
`notebooks/metro_experiments.ipynb`. Описание проведённой работы: `docs/research/README.md`.
Для оформления и публикации проекта не требуется запускать новое обучение.

## Публикация

Создай пустой GitHub-репозиторий `MetroYOLO` без автоматически создаваемого README.
Архив для публикации содержит только текущие файлы. Для загрузки через Git
создай новую историю из распакованной папки:

```bash
git init -b main
git add .
git commit -m "Initial MetroYOLO research project"
git remote add origin https://github.com/aniantm05-hash/MetroYOLO.git
git push -u origin main
```

GitHub Description: `Passenger counting with YOLOv5: transfer learning, error analysis and Streamlit dashboard`.
Темы: `computer-vision`, `yolov5`, `pytorch`, `object-detection`, `streamlit`, `machine-learning`.
