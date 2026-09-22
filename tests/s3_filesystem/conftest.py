"""Shared fixtures for S3Filesystem tests."""

from __future__ import annotations

import importlib.util

_HAS_BOTO3 = importlib.util.find_spec('boto3') is not None
collect_ignore = [] if _HAS_BOTO3 else ['test_s3_filesystem.py']
