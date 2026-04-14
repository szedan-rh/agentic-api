"""
Relational persistence layer for agentic_api.

Tables follow ADR-02: Item, Response, Conversation.

Each table module exposes:
- A SQLAlchemy ORM declarative model (the table definition)
- CRUD functions decorated with session operation decorators

The shared declarative base lives here so all table classes share the same metadata.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
