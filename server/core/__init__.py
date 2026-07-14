"""Core infrastructure package."""

from server.core.settings import settings
from server.core.database import get_connection, transaction, init_db
from server.core.json import encode, decode, decode_list, decode_dict
from server.core.exceptions import (
    AppException,
    NotFoundError,
    ValidationError,
    ConflictError,
    register_exception_handlers,
)
from server.core.utils import clamp, utcnow, radius_for, stable_position

__all__ = [
    "settings",
    "get_connection",
    "transaction",
    "init_db",
    "encode",
    "decode",
    "decode_list",
    "decode_dict",
    "AppException",
    "NotFoundError",
    "ValidationError",
    "ConflictError",
    "register_exception_handlers",
    "clamp",
    "utcnow",
    "radius_for",
    "stable_position",
]