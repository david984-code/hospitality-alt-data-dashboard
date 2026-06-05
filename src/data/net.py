"""Shared retry policy for flaky network fetches (exponential backoff)."""

from __future__ import annotations

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# For requests-based fetchers: retry transient network/HTTP errors, not logic errors.
network_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
)

# For yfinance (raises assorted exception types); retry broadly with backoff.
flaky_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
)
