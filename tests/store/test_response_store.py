from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_api.store.response import ResponseStore
from agentic_api.types.responses import InputMessage, OutputMessage
from agentic_api.utils.exceptions import BadInputError, ResponsesAPIError

from tests.utils import make_request as _make_request
from tests.utils import make_response as _make_response


@pytest.fixture
async def store(db_engine: AsyncEngine) -> ResponseStore:
    return ResponseStore(engine=db_engine)


@pytest.mark.anyio
async def test_get_returns_none_for_missing(store: ResponseStore) -> None:
    result = await store.get(response_id="does_not_exist")
    assert result is None


@pytest.mark.anyio
async def test_get_or_raise_raises_for_missing(store: ResponseStore) -> None:
    with pytest.raises(ResponsesAPIError) as exc_info:
        await store.get_or_raise(response_id="does_not_exist")
    assert exc_info.value.code == "previous_response_not_found"
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_put_and_get_round_trip(store: ResponseStore) -> None:
    request = _make_request(input="hello")
    response = _make_response(response_id="resp_001", text="hi there")

    await store.put_completed(
        request=request,
        hydrated_request=request,
        response=response,
    )

    stored = await store.get(response_id="resp_001")
    assert stored is not None
    assert stored.response_id == "resp_001"
    assert stored.model == "test-model"
    assert stored.previous_response_id is None
    assert len(stored.history_item_ids) > 0


@pytest.mark.anyio
async def test_put_skipped_when_store_false(store: ResponseStore) -> None:
    request = _make_request(input="hello", store=False)
    response = _make_response(response_id="resp_no_store")

    await store.put_completed(
        request=request,
        hydrated_request=request,
        response=response,
    )

    stored = await store.get(response_id="resp_no_store")
    assert stored is None


@pytest.mark.anyio
async def test_put_skipped_when_status_not_persistable(store: ResponseStore) -> None:
    request = _make_request(input="hello")
    response = _make_response(response_id="resp_failed")
    response.status = "failed"

    await store.put_completed(
        request=request,
        hydrated_request=request,
        response=response,
    )

    stored = await store.get(response_id="resp_failed")
    assert stored is None


@pytest.mark.anyio
async def test_duplicate_response_id_raises_bad_input(store: ResponseStore) -> None:
    request = _make_request(input="hello")
    response = _make_response(response_id="resp_dup")

    await store.put_completed(
        request=request,
        hydrated_request=request,
        response=response,
    )

    with pytest.raises(BadInputError):
        await store.put_completed(
            request=request,
            hydrated_request=request,
            response=response,
        )


@pytest.mark.anyio
async def test_previous_response_id_stored(store: ResponseStore) -> None:
    req1 = _make_request(input="turn 1")
    resp1 = _make_response(response_id="resp_t1")
    await store.put_completed(request=req1, hydrated_request=req1, response=resp1)

    req2 = _make_request(input="turn 2", previous_response_id="resp_t1")
    resp2 = _make_response(response_id="resp_t2", previous_response_id="resp_t1")
    await store.put_completed(request=req2, hydrated_request=req2, response=resp2)

    stored2 = await store.get_or_raise(response_id="resp_t2")
    assert stored2.previous_response_id == "resp_t1"


# ---------------------------------------------------------------------------
# rehydrate
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rehydrate_restores_items_in_order(store: ResponseStore) -> None:
    request = _make_request(input="hello")
    response = _make_response(response_id="resp_rh", text="world")

    await store.put_completed(
        request=request,
        hydrated_request=request,
        response=response,
    )

    stored = await store.get_or_raise(response_id="resp_rh")
    items = await store.rehydrate(stored=stored)

    # Should have: input message + output message
    assert len(items) == 2
    input_item = items[0]
    output_item = items[1]

    assert isinstance(input_item, InputMessage)
    assert input_item.role == "user"

    assert isinstance(output_item, OutputMessage)
    assert output_item.content[0].text == "world"


@pytest.mark.anyio
async def test_rehydrate_multi_turn_accumulates_history(store: ResponseStore) -> None:
    # Turn 1
    req1 = _make_request(input="first")
    resp1 = _make_response(response_id="resp_mt1", text="answer 1")
    await store.put_completed(request=req1, hydrated_request=req1, response=resp1)

    stored1 = await store.get_or_raise(response_id="resp_mt1")
    history = await store.rehydrate(stored=stored1)

    # Turn 2: hydrated_request includes full history from turn 1
    req2_base = _make_request(input="second", previous_response_id="resp_mt1")
    req2_hydrated = req2_base.model_copy(
        update={"input": [*history, InputMessage(role="user", content="second")]}
    )
    resp2 = _make_response(
        response_id="resp_mt2", text="answer 2", previous_response_id="resp_mt1"
    )
    await store.put_completed(
        request=req2_base,
        hydrated_request=req2_hydrated,
        response=resp2,
    )

    stored2 = await store.get_or_raise(response_id="resp_mt2")
    items2 = await store.rehydrate(stored=stored2)

    # Should contain: turn-1 user + turn-1 assistant + turn-2 user + turn-2 assistant
    assert len(items2) == 4
    roles = [getattr(i, "role", None) for i in items2]
    assert roles == ["user", "assistant", "user", "assistant"]
