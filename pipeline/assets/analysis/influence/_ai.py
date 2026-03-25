"""AI provider: claude CLI integration."""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import _config


def configure_ai_provider() -> None:
    """Check that the claude CLI is available and configure the module global."""
    import shutil

    if shutil.which("claude"):
        _config.set_ai_provider("claude-cli")
    else:
        _config.set_ai_provider(None)


def ai_complete(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Send a prompt to the claude CLI and return the text response."""
    if _config.AI_PROVIDER is None:
        return ""

    if json_mode:
        prompt = (
            prompt
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "Do not include any prose, markdown fences, or explanations outside the JSON."
        )

    try:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        result = subprocess.run(
            ["claude", "-p", "--model", "sonnet"],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {result.stderr[:200]}")
        return result.stdout.strip()
    except Exception:
        pass

    return ""


def parse_json_response(raw: str, retry_prompt: str = "") -> dict | list | None:
    """Parse an AI JSON response, retrying once if parsing fails."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if not retry_prompt:
        return None

    nudge = (
        f"{retry_prompt}\n\nYour previous response was not valid JSON. "
        "Respond ONLY with valid JSON, nothing else."
    )
    retry_raw = ai_complete(nudge, json_mode=True)
    if not retry_raw:
        return None

    retry_cleaned = re.sub(r"^```(?:json)?\s*", "", retry_raw.strip(), flags=re.IGNORECASE)
    retry_cleaned = re.sub(r"\s*```$", "", retry_cleaned.strip())
    try:
        return json.loads(retry_cleaned)
    except json.JSONDecodeError:
        return None


def ai_complete_parallel(
    prompts: list[str],
    *,
    system: str = "",
    json_mode: bool = False,
    label: str = "AI",
    logger: Any = None,
) -> list[str]:
    """Execute multiple AI prompts concurrently using ThreadPoolExecutor."""
    _log = logger.info if logger else print

    if _config.AI_PROVIDER is None or not prompts:
        return [""] * len(prompts)

    results: list[str] = [""] * len(prompts)
    completed = 0

    def _call(idx: int, prompt: str) -> tuple[int, str]:
        return idx, ai_complete(prompt, system=system, json_mode=json_mode)

    with ThreadPoolExecutor(max_workers=_config.AI_MAX_WORKERS) as executor:
        futures = {executor.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            try:
                idx, response = future.result()
                results[idx] = response
            except Exception as exc:
                idx = futures[future]
                if logger:
                    logger.warning(f"[{label}] Call {idx} failed: {exc}")
            completed += 1
            if completed % 20 == 0 or completed == len(prompts):
                _log(f"[{label}] {completed}/{len(prompts)} done")

    return results
