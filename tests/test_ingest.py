import pytest

from src.ingest import chunk_text


def test_chunk_sizes_and_overlap():
    chunks = chunk_text("word " * 800, size=350, overlap=50)
    assert [len(c.split()) for c in chunks] == [350, 350, 200]


def test_short_text_gives_one_chunk():
    assert len(chunk_text("just a few words here", size=350, overlap=50)) == 1


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("some text", size=10, overlap=10)
