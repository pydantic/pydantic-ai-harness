"""Xberg capability: document extraction (text, tables, metadata) through a Xberg API server.

Needs no extra beyond this package: the server is reached over HTTP.
"""

from pydantic_ai_harness.xberg._capability import Xberg
from pydantic_ai_harness.xberg._toolset import (
    XBERG_DEFAULT_URL,
    XBERG_TOOL_NAMES,
    OutputFormat,
    XbergDocument,
    XbergExtraction,
    XbergExtractionError,
    XbergToolset,
)

__all__ = [
    'XBERG_DEFAULT_URL',
    'XBERG_TOOL_NAMES',
    'OutputFormat',
    'Xberg',
    'XbergDocument',
    'XbergExtraction',
    'XbergExtractionError',
    'XbergToolset',
]
