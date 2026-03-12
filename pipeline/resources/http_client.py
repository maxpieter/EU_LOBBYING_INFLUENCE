"""Shared HTTP client resource with rate limiting for external APIs."""

import time
from threading import Lock
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from dagster import ConfigurableResource, InitResourceContext
from pydantic import Field, PrivateAttr

_GLOBAL_LAST_REQUEST_TIME: Dict[str, float] = {}
_GLOBAL_RATE_LIMIT_LOCK = Lock()


class HttpClientResource(ConfigurableResource):
    """Shared HTTP client with rate limiting and connection pooling."""

    rate_limit_delay: float = Field(default=0.75)
    eurlex_delay: float = Field(default=10.0)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=30)
    user_agent: str = Field(
        default="EULobbyInfluenceBot/1.0 (thesis research) Mozilla/5.0",
    )

    _session: Optional[requests.Session] = PrivateAttr(default=None)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    def setup_for_execution(self, context: InitResourceContext) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
            }
        )

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> requests.Response:
        if not self._session:
            raise RuntimeError("HTTP client not initialized.")

        domain = urlparse(url).netloc

        with _GLOBAL_RATE_LIMIT_LOCK:
            if domain in _GLOBAL_LAST_REQUEST_TIME:
                elapsed = time.time() - _GLOBAL_LAST_REQUEST_TIME[domain]
                required_delay = (
                    self.eurlex_delay if "eur-lex.europa.eu" in domain else self.rate_limit_delay
                )
                if elapsed < required_delay:
                    time.sleep(required_delay - elapsed)
            _GLOBAL_LAST_REQUEST_TIME[domain] = time.time()

        timeout = timeout or self.timeout
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(
                    url, params=params, headers=headers, timeout=timeout, **kwargs,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        raise last_exception

    @property
    def session(self) -> requests.Session:
        if not self._session:
            raise RuntimeError("HTTP client not initialized.")
        return self._session

    def teardown_after_execution(self, context: InitResourceContext) -> None:
        if self._session:
            self._session.close()
            self._session = None
