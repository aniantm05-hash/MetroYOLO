import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from metroyolo import training as runner

ROOT = Path(__file__).resolve().parents[1]


def test_original_protocol_and_distinct_run_names():
    args = runner.parser().parse_args([])
    assert (args.epochs, args.batch, args.image_size, args.seed) == (30, 16, 640, 0)
    cmd = runner.training_command(
        Path("repo"), Path("data.yaml"), Path("base.pt"), Path("out"), 30, 16, 640, "0", 0
    )
    assert cmd[cmd.index("--name") + 1] == "metro_yolov5s_30"
    assert cmd[cmd.index("--optimizer") + 1] == "SGD"


def test_independent_training_and_separate_checkpoints(monkeypatch, tmp_path):
    data = tmp_path / "data"
    for split, color in [("train", "red"), ("valid", "blue")]:
        (data / split / "images").mkdir(parents=True)
        (data / split / "labels").mkdir()
        Image.new("RGB", (32, 32), color).save(data / split / "images/a.png")
        (data / split / "labels/a.txt").write_text("0 0.5 0.5 0.5 0.5")
    (data / "data.yaml").touch()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").touch()
    weights = tmp_path / "base.pt"
    weights.touch()
    output = tmp_path / "results"
    args = runner.parser().parse_args(
        [
            "--dataset",
            str(data),
            "--repo",
            str(repo),
            "--weights",
            str(weights),
            "--output",
            str(output),
            "--also-10-epochs",
        ]
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *a, **kw: "70b964b6d5067fff621f724c85d0e39e6b4c8e4e",
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), __version__="test"),
    )
    commands, checkpoints = [], []

    def train(cmd, log):
        commands.append(cmd)
        best = (
            Path(cmd[cmd.index("--project") + 1]) / cmd[cmd.index("--name") + 1] / "weights/best.pt"
        )
        best.parent.mkdir(parents=True)
        best.touch()

    def evaluate(checkpoint, *a):
        checkpoints.append(checkpoint)
        return {"summary": {"count_mae": 1.0}}, object()

    monkeypatch.setattr(runner, "run_training", train)
    monkeypatch.setattr(runner, "evaluate_checkpoint", evaluate)
    runner.run(args)
    assert [c[c.index("--epochs") + 1] for c in commands] == ["30", "10"]
    assert all(c[c.index("--weights") + 1] == str(weights) for c in commands)
    assert checkpoints[1] != checkpoints[2]
    assert (output / "comparison.csv").is_file()


def test_original_notebook_is_preserved_and_training_evidence_matches():
    import json

    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/metro_experiments.ipynb", as_version=4)
    nbformat.validate(notebook)
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert "--epochs 30" in sources
    assert "search_df_sorted" in sources
    evidence = json.loads((ROOT / "experiments/notebook_evidence.json").read_text())
    assert evidence["confirmed_training"]["epochs"] == 30
    assert evidence["confirmed_training"]["seed"] == 0
    assert evidence["sweep"]["configurations"] == 350
