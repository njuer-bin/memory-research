from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def fingerprint(*parts: Any) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AddMessage(BaseModel):
    role: str
    content: str = Field(min_length=1)
    timestamp: Optional[int] = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    messages: list[AddMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    # AML Add success contract: return all request identity fields unchanged.
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    # Keep internal optional controls for local regression tests while matching
    # the official AML-required fields: query, user_id and top_k.
    model_config = ConfigDict(extra="allow")
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1, le=100)
    options: Optional[dict[str, Any]] = None

    # Internal/backward-compatible controls. AML does not need to send these.
    question: Optional[str] = None
    session_id: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    memory_types: Optional[list[str]] = None
    include_history: bool = False
    multi_hop: Optional[bool] = None


class SearchResult(BaseModel):
    # Exact public Search result item fields required by AML.
    id: str
    content: str
    score: float
    created_at: str


class SearchResponse(BaseModel):
    # Exact AML Search response envelope.
    data: list[SearchResult]
