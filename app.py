"""Local image dashboard. Demo and actual inference are explicitly separated."""

import hashlib
import io
import json
from pathlib import Path
from statistics import mean

import streamlit as st
from PIL import Image, ImageOps

from metroyolo.domain import InferenceConfig, occupancy
from metroyolo.visualization import annotate

st.set_page_config(page_title="MetroYOLO | Passenger analytics", page_icon="🚇", layout="wide")
st.markdown(
    """<style>
.block-container{max-width:1250px;padding-top:2rem}
[data-testid="stMetric"]{background:#152233;border:1px solid #26374d;
padding:18px;border-radius:12px}
[data-testid="stMetricValue"]{color:#5de5bc}
h1{letter-spacing:-1.6px} footer{visibility:hidden}
</style>""",
    unsafe_allow_html=True,
)
st.caption("COMPUTER VISION / YOLOv5 / APPLIED EVALUATION")
st.title("MetroYOLO")
st.markdown("Подсчёт пассажиров и оценка загрузки **видимой области** вагона по фотографии.")


@st.cache_resource
def load_detector(repo, weights, device):
    from metroyolo.detector import YoloV5Detector

    return YoloV5Detector(Path(repo), Path(weights), device)


with st.sidebar:
    st.header("Параметры")
    mode = st.radio("Источник результата", ["Примеры из отчёта", "YOLOv5: своё фото"])
    capacity = st.number_input("Условная вместимость видимой области", min_value=1, value=20)
    confidence = st.slider("Порог уверенности", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("IoU для NMS", 0.05, 1.0, 0.45, 0.05)
    max_det = st.number_input("Максимум детекций", min_value=1, value=100)
    st.caption("Загрузка = число людей / условная вместимость. Это не заполненность всего вагона.")

tab_research, tab_demo = st.tabs(["Результаты эксперимента", "Детекция на фото"])
with tab_demo:
    detections = None
    if mode == "Примеры из отчёта":
        st.caption(
            "Исходные иллюстрации практики. Рамки не изменены; новый inference не выполняется."
        )
        for column, name, count in zip(
            st.columns(3), ["detection-11", "detection-14", "detection-2"], [11, 14, 2], strict=True
        ):
            column.image(
                str(Path(__file__).resolve().parent / f"docs/assets/report/{name}.png"),
                caption=f"{count} детекций · исходный отчёт, стр. 35–36",
                use_container_width=True,
            )
        st.caption(
            "Настройки слева действуют при обработке своего фото "
            "и не меняют сохранённые иллюстрации."
        )
    else:
        repo = st.sidebar.text_input("Каталог YOLOv5", "vendor/yolov5")
        weights = st.sidebar.text_input("Локальные веса", "weights/yolov5s.pt")
        device = st.sidebar.selectbox("Устройство", ["cpu", "cuda:0"])
        uploaded = st.file_uploader(
            "Загрузи JPG, PNG или WebP", type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded is None:
            st.caption("Сначала выполни настройку модели из README. На Mac начни с CPU.")
        else:
            signature = (
                hashlib.sha256(uploaded.getvalue()).hexdigest(),
                repo,
                weights,
                device,
                confidence,
                iou,
                int(max_det),
            )
            if st.button("Найти пассажиров", type="primary"):
                try:
                    with st.spinner("Обработка изображения…"):
                        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                        settings = InferenceConfig(confidence, iou, int(max_det))
                        found = load_detector(repo, weights, device).predict(image, settings)
                        st.session_state["prediction"] = (signature, image, found)
                except (ValueError, OSError, ImportError, RuntimeError) as exc:
                    st.session_state.pop("prediction", None)
                    st.error(str(exc))
            saved = st.session_state.get("prediction")
            if saved and saved[0] == signature:
                _, image, detections = saved
                provenance = "yolov5"
            elif saved:
                st.caption("Параметры или фотография изменились. Запусти детекцию повторно.")
    if detections is not None:
        count = len(detections)
        load = occupancy(count, int(capacity))
        a, b, c = st.columns(3)
        a.metric("Людей в кадре", count)
        b.metric("Индекс загрузки", f"{load:.1f}%")
        c.metric(
            "Средняя уверенность",
            f"{mean(d.confidence for d in detections):.2f}" if detections else "—",
        )
        left, right = st.columns([3, 1])
        annotated = annotate(image, detections)
        left.image(annotated, caption="Видимая область · person", use_container_width=True)
        with right:
            st.subheader("Результат")
            st.progress(min(load / 100, 1.0))
            if load > 100:
                st.caption("Количество выше условной вместимости; исходный индекс не обрезается.")
            st.caption("Уверенность детектора не является измеренной точностью подсчёта.")
            payload = {
                "source": provenance,
                "count": count,
                "capacity": int(capacity),
                "visible_area_load_percent": load,
                "config": {"confidence": confidence, "iou": iou, "max_det": int(max_det)},
                "detections": [d.to_dict() for d in detections],
            }
            st.download_button(
                "Скачать JSON", json.dumps(payload, indent=2), "prediction.json", "application/json"
            )
            buffer = io.BytesIO()
            annotated.save(buffer, format="PNG")
            st.download_button(
                "Скачать изображение", buffer.getvalue(), "annotated.png", "image/png"
            )
            if detections:
                st.dataframe(
                    [
                        {"№": i, "confidence": round(d.confidence, 3)}
                        for i, d in enumerate(detections, 1)
                    ],
                    hide_index=True,
                )
with tab_research:
    root = Path(__file__).resolve().parent
    saved = json.loads((root / "experiments/baseline_rerun/summary.json").read_text())
    measured = saved["summary"]
    st.subheader("Дообучение не обязано улучшать прикладную задачу")
    st.markdown("""Проверялась адаптация YOLOv5s к фотографиям метро.
Четыре проверки: готовая модель, 30 эпох, короткий запуск на 10 эпохах
и 350 комбинаций confidence / NMS IoU / max_det.
""")
    st.image(str(root / "docs/assets/reported-comparison.png"), use_container_width=True)
    st.caption(
        "1,90 — агрегат исходного отчёта, стр. 31. Baseline повторно проверен; "
        "исторические дообученные веса не сохранились."
    )
    st.markdown("""**Подготовка данных.** 81 train / 21 validation, один класс person.
Авторазметка YOLOv5s корректировалась вручную в Roboflow; 10 кадров без людей
сохранены как отрицательные примеры. Baseline помогал создавать эталон:
это возможный источник преимущества при сравнении моделей.
""")
    st.subheader("Что происходило при обучении")
    st.image(str(root / "docs/assets/training-loss.png"), use_container_width=True)
    st.image(str(root / "docs/assets/training-history.png"), use_container_width=True)
    st.markdown("""Box loss уменьшается, detection-метрики растут с колебаниями.
Но они не заменяют ошибку количества людей. Class loss = 0 относится к устройству
одноклассового запуска, а не к идеальной точности.
""")
    st.subheader("Помогло ли обучать меньше или менять пороги?")
    st.markdown("""**Короткий запуск:** в отчёте описаны лишние рамки на поручнях,
руках и сиденьях. В выводе 10EPOCH сохранены MAE 28,62 и среднее предсказание
36,19 против 7,57 по разметке. Путь весов в notebook переиспользован командой
на 30 эпох; точная привязка checkpoint не подтверждена.

**Постобработка:** 10 confidence × 7 IoU × 5 max_det = 350 настроек.
По отчёту, ни одна не превзошла baseline. Рейтинг не сохранился;
тепловая карта и лучший порог не выдуманы. Ограничить число рамок — ещё не значит
исправить детекции.
""")
    st.subheader("Почему среднего балла недостаточно")
    a, b, c = st.columns(3)
    a.metric("Baseline count MAE", f"{measured['count_mae']:.2f} человека")
    b.metric("Точный подсчёт", "10 из 21")
    c.metric("Ошибка не больше двух", "19 из 21")
    st.image(str(root / "docs/assets/baseline-analysis.png"), use_container_width=True)
    st.markdown("""В худшем кадре найдены 11 вместо 16 человек.
Средний confidence найденных рамок — 0,56, но пропущенные люди в него не попадают.
И наоборот: один пропуск и одна лишняя рамка могут дать правильное количество.
Поэтому нужно анализировать рамки и FP/FN вместе с count MAE.

**Решение исходной работы:** оставить pretrained YOLOv5s.
Возможные причины ухудшения — малый датасет, шум меток и teacher bias.
Для их проверки нужен независимый test по сценам и сохранённые checkpoint,
а не просто больше эпох.
""")
    st.download_button(
        "Скачать результаты baseline по всем кадрам",
        (root / "experiments/baseline_rerun/per_image.csv").read_bytes(),
        "baseline_per_image.csv",
        "text/csv",
    )
    st.caption("Полная история и источники: docs/research/ и docs/SOURCE_MAP.md.")
