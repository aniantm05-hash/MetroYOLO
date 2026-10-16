.PHONY: demo app test lint setup

demo:
	python -m metroyolo.cli demo
app:
	python -m streamlit run app.py
setup:
	bash scripts/setup_yolov5.sh
test:
	python -m pytest -q
lint:
	python -m ruff check .
