"""
Smoke tests that need neither Airflow, Docker nor the Kaggle dataset.

Run locally:   uv run --group dev pytest -q
Run in CI:     see .github/workflows/ci.yml (plain pip install of the deps)
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def test_config_loads_and_has_pipeline_sections():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    for key in ("data", "data_cleaning", "feature_engineering", "model", "cross_validation", "mlflow"):
        assert key in cfg, f"config.yaml missing top-level section {key!r}"
    assert cfg["model"]["type"] == "lightgbm"


@pytest.mark.parametrize("dag_file", sorted((REPO / "dags").glob("*.py")), ids=lambda p: p.name)
def test_dag_files_are_valid_python(dag_file: Path):
    # Full DAG import needs an Airflow install; a syntax pass still catches the
    # most common breakage (bad merges, stray characters, Windows-only edits).
    ast.parse(dag_file.read_text(), filename=str(dag_file))


def test_no_windows_paths_in_source():
    offenders = []
    for py in list((REPO / "src").glob("*.py")) + list((REPO / "scripts").glob("*.py")):
        if "C:\\\\" in py.read_text() or "/mnt/c/" in py.read_text():
            offenders.append(py.name)
    assert not offenders, f"hard-coded Windows/WSL paths in {offenders}"


def test_pipeline_end_to_end_on_synthetic_data(tmp_path: Path, monkeypatch):
    """ingestion -> cleaning -> feature engineering -> training on generated data."""
    from make_smoke_data import make_smoke_data
    from data_ingestion import run_data_ingestion
    from data_cleaning import run_data_cleaning
    from feature_engineering import run_feature_engineering
    from model_training import run_model_training

    raw = tmp_path / "unzipped"
    make_smoke_data(raw, n_rows=600)

    # Re-point every path in config.yaml at the temp dir and disable MLflow.
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    d = cfg["data"]
    d["raw_data_dir"] = str(raw)
    d["processed_data_dir"] = str(tmp_path / "processed")
    d["output_dir"] = str(tmp_path / "output")
    d["models_dir"] = str(tmp_path / "models")
    for name in ("train_transaction", "train_identity", "test_transaction", "test_identity"):
        d[name] = str(raw / f"{name}.csv")
    cfg["model"]["lightgbm"]["n_estimators"] = 20
    cfg["model"]["early_stopping_rounds"] = 5
    cfg["mlflow"]["tracking_uri"] = ""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    ing = run_data_ingestion(config_path=str(cfg_path), run_id="smoke")
    cln = run_data_cleaning(config_path=str(cfg_path), run_id="smoke", upstream_manifest_path=ing["manifest_path"])
    fe = run_feature_engineering(config_path=str(cfg_path), run_id="smoke", upstream_manifest_path=cln["manifest_path"])
    tr = run_model_training(config_path=str(cfg_path), run_id="smoke", upstream_manifest_path=fe["manifest_path"])

    assert Path(tr["manifest_path"]).exists()
    assert 0.0 <= tr["overall_roc_auc"] <= 1.0
    assert Path(tr["test_predictions_path"]).stat().st_size > 0

    _assert_streamlit_preprocessor_matches_pipeline(ing, cln, fe)


def _assert_streamlit_preprocessor_matches_pipeline(ing, cln, fe):
    """The artifacts written by feature engineering must let the Streamlit
    preprocessor rebuild the exact rows the model was trained on."""
    import json
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(REPO / "streamlit_app"))
    from preprocessing import FraudPreprocessor

    manifest = json.loads(Path(fe["manifest_path"]).read_text())
    artifacts_dir = Path(manifest["outputs"]["feature_names_path"]).parent
    pre = FraudPreprocessor(str(artifacts_dir))

    X_train = pd.read_csv(fe["X_train_path"])
    assert pre.feature_names == list(X_train.columns)

    raw = pd.read_csv(ing["train_path"]).set_index("TransactionID")
    cleaned = pd.read_csv(cln["train_path"]).sort_values("TransactionDT")
    for i, tx_id in enumerate(cleaned["TransactionID"].head(25)):
        got = pre.preprocess(raw.loc[tx_id].rename_axis(None).copy())[0]
        want = X_train.iloc[i].to_numpy(dtype=np.float32)
        assert got.shape == want.shape
        same = np.isclose(got, want, rtol=1e-4, equal_nan=True)
        bad = [f"{n}: got {g}, want {w}" for n, g, w, ok in zip(pre.feature_names, got, want, same) if not ok]
        assert not bad, f"TransactionID {tx_id} differs in {bad}"


def test_data_cleaning_local_fallback_finds_latest_ingestion(tmp_path: Path):
    """Regression test for the removed hard-coded C:\\ path in data_cleaning."""
    from make_smoke_data import make_smoke_data
    from data_ingestion import run_data_ingestion
    from data_cleaning import run_data_cleaning

    raw = tmp_path / "unzipped"
    make_smoke_data(raw, n_rows=200)
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    d = cfg["data"]
    d["processed_data_dir"] = str(tmp_path / "processed")
    for name in ("train_transaction", "train_identity", "test_transaction", "test_identity"):
        d[name] = str(raw / f"{name}.csv")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    import json

    run_data_ingestion(config_path=str(cfg_path), run_id="first")
    out = run_data_cleaning(config_path=str(cfg_path))  # no paths given -> must auto-discover
    manifest = json.loads(Path(out["manifest_path"]).read_text())
    assert "/ingestion/first/" in manifest["inputs"]["train_path"]
    assert Path(out["train_path"]).exists()


def test_k8s_retraining_dag_has_a_real_schedule():
    """Regression test: this is the scheduled-retraining DAG (see check_new_data.py
    for the data-changed gate). If schedule/schedule_interval regresses to None,
    Airflow never triggers it on its own and it silently goes back to being a
    manually-clicked training script."""
    dag_file = REPO / "dags" / "fraud_detection_dag_k8s.py"
    tree = ast.parse(dag_file.read_text(), filename=str(dag_file))

    dag_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DAG"
    ]
    assert dag_calls, f"no DAG(...) call found in {dag_file.name}"

    schedule_kwargs = [
        kw for call in dag_calls for kw in call.keywords
        if kw.arg in ("schedule", "schedule_interval")
    ]
    assert schedule_kwargs, f"{dag_file.name}: DAG(...) call has no schedule/schedule_interval kwarg"
    for kw in schedule_kwargs:
        is_none_literal = isinstance(kw.value, ast.Constant) and kw.value.value is None
        assert not is_none_literal, f"{dag_file.name}: schedule is None -- DAG never runs on its own"


def _write_smoke_raw_csvs(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train_transaction.csv", "train_identity.csv", "test_transaction.csv", "test_identity.csv"):
        (raw_dir / name).write_text("TransactionID,isFraud\n1,0\n")


def test_check_new_data_detects_new_and_unchanged_data(tmp_path: Path, monkeypatch):
    import check_new_data

    raw_dir = tmp_path / "unzipped"
    state_file = tmp_path / ".last_trained_fingerprint.json"
    _write_smoke_raw_csvs(raw_dir)

    monkeypatch.setattr(check_new_data, "RAW_DIR", raw_dir)
    monkeypatch.setattr(check_new_data, "STATE", state_file)

    # First run: no prior fingerprint saved yet -> proceed.
    monkeypatch.setattr(sys, "argv", ["check_new_data.py"])
    assert check_new_data.main() == 0

    # Commit the fingerprint (as the DAG does after a successful training run),
    # then check again with nothing changed -> skip.
    monkeypatch.setattr(sys, "argv", ["check_new_data.py", "--commit"])
    assert check_new_data.main() == 0
    assert state_file.exists()

    monkeypatch.setattr(sys, "argv", ["check_new_data.py"])
    assert check_new_data.main() == check_new_data.SKIP_CODE

    # Touching a raw file (mtime change, same content) counts as new data.
    target = raw_dir / "train_transaction.csv"
    old_mtime = target.stat().st_mtime
    os.utime(target, (old_mtime + 5, old_mtime + 5))
    assert check_new_data.main() == 0

    # A missing raw file must fail hard, not be mistaken for "skip".
    target.unlink()
    with pytest.raises(SystemExit) as exc_info:
        check_new_data.main()
    assert exc_info.value.code not in (0, check_new_data.SKIP_CODE)
