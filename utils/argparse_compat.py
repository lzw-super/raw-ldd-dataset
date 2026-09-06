"""Compatibility helpers for command-line parsing across supported Python versions."""

from __future__ import annotations

import argparse
from typing import Any


def add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    default: bool | None,
    **kwargs: Any,
) -> None:
    """Add ``--flag``/``--no-flag`` on Python 3.8 and newer.

    ``argparse.BooleanOptionalAction`` was added in Python 3.9.  The project
    runs in its pinned Python 3.8 environment as well, so reproduce its
    behaviour with a mutually exclusive pair when the native action is absent.
    """
    native_action = getattr(argparse, "BooleanOptionalAction", None)
    if native_action is not None:
        parser.add_argument(option, action=native_action, default=default, **kwargs)
        return

    if not option.startswith("--"):
        raise ValueError(f"Boolean optional argument must be a long option, got {option!r}")

    destination = option[2:].replace("-", "_")
    negative_option = f"--no-{option[2:]}"
    group = parser.add_mutually_exclusive_group()
    group.add_argument(option, dest=destination, action="store_true", **kwargs)
    group.add_argument(negative_option, dest=destination, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{destination: default})
