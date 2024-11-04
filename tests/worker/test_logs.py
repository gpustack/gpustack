from typing import List, Union
import pytest

from gpustack.worker.logs import LogOptions, log_generator


@pytest.fixture
def sample_log_file(tmp_path):
    log_content = "line1\nline2\nline3\nline4\nline5\n"
    log_file = tmp_path / "test.log"
    log_file.write_text(log_content)
    return log_file


@pytest.fixture
def large_log_file(tmp_path):
    # Create a log file with 2KB in two lines
    log_content = "line" * 256 + "\n" + "line" * 256 + "\n"
    log_file = tmp_path / "large_test.log"
    log_file.write_text(log_content)
    return log_file


def normalize_newlines(data: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(data, str):
        return data.replace("\r\n", "\n")
    elif isinstance(data, list):
        return [line.replace("\r\n", "\n") for line in data]


def test_log_generator_default(sample_log_file):
    options = LogOptions()
    log_path = str(sample_log_file)

    result = normalize_newlines(list(log_generator(log_path, options)))
    assert result == [
        "line1\n",
        "line2\n",
        "line3\n",
        "line4\n",
        "line5\n",
    ]


def test_log_generator_tail(sample_log_file):
    options = LogOptions(tail=2)
    log_path = str(sample_log_file)

    result = normalize_newlines(list(log_generator(log_path, options)))
    assert result == ["line4\n", "line5\n"]


def test_log_generator_follow(sample_log_file):
    options = LogOptions(follow=True)
    log_path = str(sample_log_file)

    generator = log_generator(log_path, options)
    result = normalize_newlines([next(generator) for _ in range(5)])
    assert result == ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]

    # Append a new line to the log file
    with open(log_path, "a") as file:
        file.write("line6\n")
    result = normalize_newlines(next(generator))
    assert result == "line6\n"


def test_log_generator_empty_file(tmp_path):
    empty_file = tmp_path / "empty.log"
    empty_file.touch()
    options = LogOptions(tail=0)

    result = list(log_generator(str(empty_file), options))
    assert result == []


def test_log_generator_tail_larger_than_file(sample_log_file):
    options = LogOptions(tail=10)
    log_path = str(sample_log_file)

    result = normalize_newlines(list(log_generator(log_path, options)))
    assert result == ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]


def test_log_generator_tail_large_file(large_log_file):
    options = LogOptions(tail=1)
    log_path = str(large_log_file)

    result = normalize_newlines(list(log_generator(log_path, options)))
    assert result == ["line" * 256 + "\n"]


def test_log_generator_tail_larger_than_large_file(large_log_file):
    options = LogOptions(tail=3)
    log_path = str(large_log_file)

    result = normalize_newlines(list(log_generator(log_path, options)))
    assert result == ["line" * 256 + "\n", "line" * 256 + "\n"]
