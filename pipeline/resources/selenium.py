"""Selenium browser pool resource for Dagster."""

import queue
import random
import threading
import time
from typing import Optional

from dagster import ConfigurableResource
from pydantic import Field
from selenium import webdriver


class BrowserPool:
    """Manages a pool of Chrome browser instances for efficient scraping."""

    def __init__(self, pool_size: int = 3, page_load_timeout: int = 30, implicit_wait: int = 10):
        self.pool_size = pool_size
        self.page_load_timeout = page_load_timeout
        self.implicit_wait = implicit_wait
        self.drivers: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self._initialize_pool()

    def _create_driver(self) -> webdriver.Chrome:
        """Create a new Chrome driver instance."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-logging")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-tools")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-translate")
        options.add_argument("--mute-audio")
        options.add_argument("--no-first-run")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        )
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(self.page_load_timeout)
        driver.implicitly_wait(self.implicit_wait)

        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )

        return driver

    def _initialize_pool(self):
        """Initialize the browser pool."""
        for _ in range(self.pool_size):
            try:
                driver = self._create_driver()
                self.drivers.put(driver)
            except Exception as e:
                print(f"Failed to create browser instance: {e}")

    def get_driver(self) -> webdriver.Chrome:
        """Get a driver from the pool, create new one if pool is empty."""
        try:
            driver = self.drivers.get_nowait()
            try:
                driver.get("about:blank")
            except Exception:
                pass
            return driver
        except queue.Empty:
            return self._create_driver()

    def return_driver(self, driver: webdriver.Chrome):
        """Return a driver to the pool."""
        if driver and self.drivers.qsize() < self.pool_size:
            try:
                driver.get("about:blank")
                driver.delete_all_cookies()
                self.drivers.put(driver)
            except Exception:
                self._safe_quit(driver)
        else:
            self._safe_quit(driver)

    def _safe_quit(self, driver: Optional[webdriver.Chrome]):
        """Safely quit a driver."""
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    def close_all(self):
        """Close all drivers in the pool."""
        while not self.drivers.empty():
            try:
                driver = self.drivers.get_nowait()
                driver.quit()
            except Exception:
                pass


class SeleniumResource(ConfigurableResource):
    """Dagster resource for Selenium browser pool."""

    pool_size: int = Field(default=3, description="Number of browser instances to keep in pool")
    page_load_timeout: int = Field(default=30, description="Seconds to wait for page load")
    implicit_wait: int = Field(default=10, description="Seconds to wait for elements")
    max_retries: int = Field(default=3, description="Number of retry attempts for failed requests")

    _pool: Optional[BrowserPool] = None

    def get_pool(self) -> BrowserPool:
        """Get or create the browser pool."""
        if self._pool is None:
            self._pool = BrowserPool(
                pool_size=self.pool_size,
                page_load_timeout=self.page_load_timeout,
                implicit_wait=self.implicit_wait,
            )
        return self._pool

    def get_driver(self) -> webdriver.Chrome:
        """Get a driver from the pool."""
        return self.get_pool().get_driver()

    def return_driver(self, driver: webdriver.Chrome):
        """Return a driver to the pool."""
        self.get_pool().return_driver(driver)

    def fetch_with_retry(self, url: str, wait_time: float = 1.0) -> str:
        """Fetch a URL with retry logic."""
        pool = self.get_pool()

        for attempt in range(self.max_retries):
            driver = None
            try:
                driver = pool.get_driver()
                throttle_delay = random.uniform(1, 3)
                time.sleep(throttle_delay)
                driver.get(url)
                page_source = driver.page_source
                pool.return_driver(driver)
                driver = None
                return page_source
            except Exception as e:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                if attempt < self.max_retries - 1:
                    sleep_time = (2**attempt) + random.uniform(1, 3)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(
                        f"Failed to fetch {url} after {self.max_retries} attempts: {e}"
                    )

        raise RuntimeError(f"Failed to fetch {url}")

    def teardown(self):
        """Clean up resources."""
        if self._pool:
            self._pool.close_all()
            self._pool = None
