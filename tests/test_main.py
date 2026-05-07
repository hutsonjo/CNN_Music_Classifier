from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from music_classifier.cli.main import main, valid_audio_file


# ---------------------------------------------------------------------------
# valid_audio_file
# ---------------------------------------------------------------------------


def test_valid_audio_file_returns_path_for_existing_file(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("dummy audio content")

    result = valid_audio_file(str(audio_file))

    assert result == audio_file


def test_valid_audio_file_raises_for_missing_file() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
        valid_audio_file("missing.wav")


def test_valid_audio_file_raises_for_directory(tmp_path: Path) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="is not a file"):
        valid_audio_file(str(tmp_path))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_success_prints_predictions(mocker, capsys) -> None:
    fake_args = argparse.Namespace(audio_file=Path("song.wav"), top_n=2)
    fake_model = mocker.Mock()
    fake_results = [
        ("rock", 0.721),
        ("metal", 0.201),
    ]

    mocker.patch(
        "music_classifier.cli.main.build_parser",
        return_value=fake_args,
    )
    load_model_mock = mocker.patch(
        "music_classifier.cli.main.load_genre_model",
        return_value=fake_model,
    )
    classify_file_mock = mocker.patch(
        "music_classifier.cli.main.classify_file",
        return_value=fake_results,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    load_model_mock.assert_called_once_with()
    classify_file_mock.assert_called_once_with(
        fake_model,
        fake_args.audio_file,
        fake_args.top_n,
    )

    captured = capsys.readouterr()
    assert "Predictions:" in captured.out
    assert "rock" in captured.out
    assert "0.721" in captured.out
    assert "metal" in captured.out
    assert "0.201" in captured.out


def test_main_success_passes_none_top_n(mocker, capsys) -> None:
    fake_args = argparse.Namespace(audio_file=Path("song.wav"), top_n=None)
    fake_model = mocker.Mock()
    fake_results = [
        ("rock", 0.721),
        ("metal", 0.201),
        ("jazz", 0.050),
    ]

    mocker.patch(
        "music_classifier.cli.main.build_parser",
        return_value=fake_args,
    )
    mocker.patch(
        "music_classifier.cli.main.load_genre_model",
        return_value=fake_model,
    )
    classify_file_mock = mocker.patch(
        "music_classifier.cli.main.classify_file",
        return_value=fake_results,
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    classify_file_mock.assert_called_once_with(
        fake_model,
        fake_args.audio_file,
        None,
    )

    captured = capsys.readouterr()
    assert "rock" in captured.out
    assert "jazz" in captured.out


def test_main_exits_with_code_1_when_model_load_fails(mocker, capsys) -> None:
    fake_args = argparse.Namespace(audio_file=Path("song.wav"), top_n=2)

    mocker.patch(
        "music_classifier.cli.main.build_parser",
        return_value=fake_args,
    )
    mocker.patch(
        "music_classifier.cli.main.load_genre_model",
        side_effect=RuntimeError("model missing"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Error: model missing" in captured.out


def test_main_exits_with_code_1_when_classification_fails(mocker, capsys) -> None:
    fake_args = argparse.Namespace(audio_file=Path("song.wav"), top_n=2)
    fake_model = mocker.Mock()

    mocker.patch(
        "music_classifier.cli.main.build_parser",
        return_value=fake_args,
    )
    mocker.patch(
        "music_classifier.cli.main.load_genre_model",
        return_value=fake_model,
    )
    mocker.patch(
        "music_classifier.cli.main.classify_file",
        side_effect=RuntimeError("bad audio"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Error: bad audio" in captured.out
