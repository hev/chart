import argparse
import tomllib
from pathlib import Path

import pytest

from chart_common.cli import non_negative_int, positive_float, positive_int
from chart_common.config import ARCTIC_QUERY_PREFIX, EMBED_DIM, FULL_CORPUS_NOTES, Settings
from functions import measure_classify_events
from indexer import measure_embed
from smoke import full_status, gates


def test_positive_int_rejects_zero_or_negative_values() -> None:
    assert positive_int("3") == 3
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        positive_int("-1")


def test_non_negative_int_allows_zero_but_rejects_negative_values() -> None:
    assert non_negative_int("0") == 0
    assert non_negative_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
        non_negative_int("-1")


def test_positive_float_rejects_zero_or_negative_values() -> None:
    assert positive_float("2.5") == 2.5
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        positive_float("0")
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        positive_float("-0.1")


def test_phase6_row_target_is_shared_across_gates_and_estimators() -> None:
    assert FULL_CORPUS_NOTES == 167_000
    assert full_status.FULL_CORPUS_NOTES == FULL_CORPUS_NOTES
    assert gates.FULL_CORPUS_NOTES == FULL_CORPUS_NOTES
    assert measure_embed.FULL_CORPUS_NOTES == FULL_CORPUS_NOTES
    assert measure_classify_events.FULL_CORPUS_NOTES == FULL_CORPUS_NOTES


def test_phase0_model_and_dataset_pins_are_exact() -> None:
    settings = Settings(_env_file=None)

    assert EMBED_DIM == 768
    assert settings.embed_model == "Snowflake/snowflake-arctic-embed-m-v1.5"
    assert ARCTIC_QUERY_PREFIX == "Represent this sentence for searching relevant passages: "
    assert settings.dataset_repo == "zhengyun21/PMC-Patients"
    assert settings.dataset_revision == "28d8836518f86d4f1e6358ea8ec09977023e5766"
    assert settings.recds_repo == "zhengyun21/PMC-Patients-ReCDS"
    assert settings.recds_revision == "a27717bb27679cf0860305997685547ca01b3dd1"
    assert len(settings.dataset_revision) == 40
    assert len(settings.recds_revision) == 40


def test_classifier_extra_keeps_vllm_linux_only_for_local_cli_inspection() -> None:
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    classifier_deps = pyproject["project"]["optional-dependencies"]["classifier"]

    assert "vllm>=0.6; sys_platform == 'linux'" in classifier_deps
