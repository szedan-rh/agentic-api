from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_api.database.conversation import (
    create_conversation,
    get_conversation,
    update_conversation_item_ids,
)
from agentic_api.database.item import Item, get_items
from agentic_api.database.response import Response
from agentic_api.database.session import configure_session_factory, session_transaction
from agentic_api.store.response import ResponseMetadata
from agentic_api.store.translator import ItemPayload
from agentic_api.types.responses import InputItem, OutputItem
from agentic_api.utils.common import utcnow, uuid7_str
from agentic_api.utils.exceptions import BadInputError, ResponsesAPIError


@dataclass(frozen=True, slots=True)
class StoredConversation:
    conversation_id: str
    history_item_ids: list[str]
    created_at: datetime
    metadata: ResponseMetadata | None = None


@session_transaction
async def _persist_conversation_turn(
    *,
    item_tuples: list[tuple[str, dict[str, Any]]],
    conversation_id: str,
    response_id: str,
    previous_response_id: str | None,
    full_item_ids: list[str],
    metadata: dict[str, Any],
) -> list[Item | Response]:
    """Atomically write new Item rows and a Response checkpoint, then update Conversation.history_item_ids.

    Item and Response are built as ORM objects and returned for a single flush by
    @session_transaction. update_conversation_item_ids uses @run_in_session which joins
    the same active session, so all writes commit together.
    """
    now = utcnow()
    items = [
        Item(id=item_id, data=data, created_at=now) for item_id, data in item_tuples
    ]
    response = Response(
        id=response_id,
        conversation_id=conversation_id,
        previous_response_id=previous_response_id,
        history_item_ids=full_item_ids,
        metadata_=metadata,
        created_at=now,
        updated_at=now,
    )
    await update_conversation_item_ids(id=conversation_id, item_ids=full_item_ids)
    return [*items, response]


class ConversationStore:
    """Read/write interface between the Conversation API layer and the three-table DB schema.

    create        — inserts a new Conversation row with a server-generated ID.
    get_or_create — load an existing Conversation by ID, or create a new one if no ID
                    is provided (or the ID is not found). Always returns a StoredConversation.
    get           — loads a Conversation row and returns a StoredConversation read model.
    put_turn      — atomically writes new Item rows, a Response checkpoint, and extends
                    Conversation.history_item_ids — all in a single Session commit.
    rehydrate     — bulk-fetches Item rows by Conversation.history_item_ids, restores order,
                    and returns the ordered history as a list of items.
    """

    def __init__(self, *, engine: AsyncEngine) -> None:
        # Session factory is shared — configure_session_factory is idempotent.
        configure_session_factory(engine)

    async def create(self) -> StoredConversation:
        """Create a new Conversation with a server-generated ID."""
        row = await create_conversation(id=uuid7_str("conv_"))
        return StoredConversation(
            conversation_id=row.id,
            history_item_ids=[],
            created_at=row.created_at,
        )

    async def get_or_create(self, *, conversation_id: str) -> StoredConversation:
        """Return an existing Conversation by ID, or create a new one if not found.

        A client may send a conversation_id it generated itself (e.g. a UUIDv7) before
        the server has ever seen it — in that case we create the row on first use.
        """
        stored = await self.get(conversation_id=conversation_id)
        if stored is not None:
            return stored

        row = await create_conversation(id=conversation_id)
        return StoredConversation(
            conversation_id=row.id,
            history_item_ids=[],
            created_at=row.created_at,
        )

    async def get(self, *, conversation_id: str) -> StoredConversation | None:
        row = await get_conversation(id=conversation_id)
        if row is None:
            return None
        return StoredConversation(
            conversation_id=row.id,
            history_item_ids=row.history_item_ids or [],
            created_at=row.created_at,
            metadata=ResponseMetadata.model_validate(row.metadata_)
            if row.metadata_
            else None,
        )

    async def put_turn(
        self,
        *,
        conversation_id: str,
        response_id: str,
        previous_response_id: str | None,
        new_items: list[InputItem | OutputItem],
        metadata_: dict[str, Any],
    ) -> StoredConversation:
        """Persist a new conversation turn atomically.

        Within a single Session commit:
        1. Bulk-insert new Item rows.
        2. Insert Response checkpoint with history_item_ids set to the full ordered list.
        3. Update Conversation.history_item_ids to append the new item IDs.

        Raises BadInputError if conversation_id does not exist or response_id already exists.
        """
        stored = await self.get(conversation_id=conversation_id)
        if stored is None:
            raise BadInputError(f"Conversation not found: {conversation_id}")

        item_tuples: list[tuple[str, dict[str, Any]]] = [
            (uuid7_str("item_"), ItemPayload(item=item).model_dump(mode="json"))
            for item in new_items
        ]
        full_item_ids = [
            *stored.history_item_ids,
            *(item_id for item_id, _ in item_tuples),
        ]

        try:
            await _persist_conversation_turn(
                item_tuples=item_tuples,
                conversation_id=conversation_id,
                response_id=response_id,
                previous_response_id=previous_response_id,
                full_item_ids=full_item_ids,
                metadata=metadata_,
            )
        except IntegrityError as e:
            raise BadInputError(f"Response id already exists: {response_id}") from e

        return StoredConversation(
            conversation_id=stored.conversation_id,
            history_item_ids=full_item_ids,
            created_at=stored.created_at,
            metadata=stored.metadata,
        )

    async def rehydrate(self, *, conversation_id: str) -> list[InputItem | OutputItem]:
        """Return the full ordered history for a conversation."""
        stored = await self.get(conversation_id=conversation_id)
        if stored is None:
            raise ResponsesAPIError(
                f"Conversation '{conversation_id}' not found.",
                status_code=400,
                param="conversation_id",
                code="conversation_not_found",
            )

        items_by_id: dict[str, Item] = {
            item.id: item for item in await get_items(ids=stored.history_item_ids)
        }
        return [
            ItemPayload.model_validate(items_by_id[item_id].data).item
            for item_id in stored.history_item_ids
            if item_id in items_by_id
        ]
