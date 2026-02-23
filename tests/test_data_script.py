from __future__ import annotations

import sys

from ai_t9._scripts.data import cmd_fetch_hf


class _FakeClient:
    def __init__(self) -> None:
        self.uploaded_part_sizes: list[int] = []
        self.completed = False
        self.aborted = False
        self.put_object_calls: list[dict] = []

    def create_multipart_upload(self, **kwargs):
        return {"UploadId": "upload-1"}

    def upload_part(self, *, Body: bytes, **kwargs):
        self.uploaded_part_sizes.append(len(Body))
        return {"ETag": f"etag-{len(self.uploaded_part_sizes)}"}

    def complete_multipart_upload(self, **kwargs):
        self.completed = True

    def abort_multipart_upload(self, **kwargs):
        self.aborted = True

    def put_object(self, **kwargs):
        self.put_object_calls.append(kwargs)


class _DatasetModule:
    def __init__(self, rows):
        self._rows = rows

    def load_dataset(self, *args, **kwargs):
        return iter(self._rows)


def test_fetch_hf_uses_fixed_size_non_final_parts(monkeypatch):
    mb = 1024 * 1024
    rows = [
        {"text": "a" * (3 * mb)},
        {"text": "b" * (7 * mb)},
        {"text": "c" * (6 * mb)},
    ]

    monkeypatch.setitem(sys.modules, "datasets", _DatasetModule(rows))

    client = _FakeClient()
    rc = cmd_fetch_hf(
        client=client,
        bucket="bucket",
        dataset="dummy",
        config="dummy",
        split="train",
        remote="corpuses/dummy.txt",
        verbose=False,
    )

    assert rc == 0
    assert client.completed
    assert not client.aborted
    assert len(client.uploaded_part_sizes) >= 2

    # All non-final parts must be exactly equal length.
    non_final = client.uploaded_part_sizes[:-1]
    assert all(size == non_final[0] for size in non_final)
    assert non_final[0] == 8 * mb


def test_fetch_hf_empty_dataset_writes_empty_object(monkeypatch):
    rows = [{"text": ""}, {"text": "   "}]
    monkeypatch.setitem(sys.modules, "datasets", _DatasetModule(rows))

    client = _FakeClient()
    rc = cmd_fetch_hf(
        client=client,
        bucket="bucket",
        dataset="dummy",
        config="dummy",
        split="train",
        remote="corpuses/empty.txt",
        verbose=False,
    )

    assert rc == 0
    assert client.aborted
    assert not client.completed
    assert client.uploaded_part_sizes == []
    assert len(client.put_object_calls) == 1
    assert client.put_object_calls[0]["Body"] == b""
