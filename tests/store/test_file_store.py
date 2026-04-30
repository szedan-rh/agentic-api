import pytest

from agentic_api.store.file_store import FileStore


@pytest.fixture
async def file_store(db_engine):
    return FileStore(engine=db_engine)


@pytest.mark.anyio
async def test_upload_and_get(file_store: FileStore):
    obj = await file_store.upload(
        filename="test.txt", purpose="assistants", content=b"hello"
    )
    assert obj.id.startswith("file-")
    assert obj.filename == "test.txt"
    assert obj.bytes == 5

    fetched = await file_store.get(file_id=obj.id)
    assert fetched is not None
    assert fetched.id == obj.id


@pytest.mark.anyio
async def test_get_content(file_store: FileStore):
    obj = await file_store.upload(
        filename="data.bin", purpose="assistants", content=b"\x00\x01\x02"
    )
    content = await file_store.get_content(file_id=obj.id)
    assert content == b"\x00\x01\x02"


@pytest.mark.anyio
async def test_list_files(file_store: FileStore):
    await file_store.upload(filename="a.txt", purpose="assistants", content=b"a")
    await file_store.upload(filename="b.txt", purpose="assistants", content=b"b")
    result = await file_store.list()
    assert len(result.data) >= 2


@pytest.mark.anyio
async def test_delete_file(file_store: FileStore):
    obj = await file_store.upload(
        filename="del.txt", purpose="assistants", content=b"delete me"
    )
    result = await file_store.delete(file_id=obj.id)
    assert result is not None
    assert result.deleted is True

    fetched = await file_store.get(file_id=obj.id)
    assert fetched is None


@pytest.mark.anyio
async def test_get_nonexistent(file_store: FileStore):
    result = await file_store.get(file_id="file-0000000000000000")
    assert result is None


def test_filename_sanitization():
    assert FileStore.sanitize_filename("../../../etc/passwd") == "passwd"
    assert FileStore.sanitize_filename("test\x00file.txt") == "testfile.txt"
    assert FileStore.sanitize_filename(".hidden") == "hidden"
    assert FileStore.sanitize_filename("") == "download"
    assert FileStore.sanitize_filename("path/to/file.txt") == "file.txt"
