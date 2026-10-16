from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_shows_historical_examples_without_fabricated_inference():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.radio[0].value == "Примеры из отчёта"
    assert len(app.metric) == 3
    assert all(m.label != "Людей в кадре" for m in app.metric)
    app.slider[0].set_value(0.9).run()
    assert not app.exception
    assert len(app.metric) == 3  # Historical figures are not recomputed by threshold controls.


def test_dashboard_real_mode_waits_for_upload():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.radio[0].set_value("YOLOv5: своё фото").run()
    assert not app.exception
    assert len(app.metric) == 3  # Research tab only; no fabricated inference.
