"""Unit tests for the utils functions."""

import tempfile

from src.utils import read_file


def test_read_file_exists_file():
    features = "rooms, zipcode, median_price, school_rating, transport"
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write(features)
    content = read_file(tmp.name)
    assert isinstance(content, str)
    assert content == features


def test_read_dont_exists_file():
    try:
        read_file("filename")
    except FileNotFoundError:
        assert True
