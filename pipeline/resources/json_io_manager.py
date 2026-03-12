"""JSON file IO manager for Dagster asset persistence."""

import json
import os
from typing import Any

from dagster import IOManager, InputContext, OutputContext


class JsonFileIOManager(IOManager):
    """Stores asset outputs as JSON files."""

    def __init__(self, base_path: str):
        self.base_path = base_path

    def _get_path(self, context) -> str:
        keys = context.asset_key.path
        if context.has_partition_key:
            return os.path.join(self.base_path, *keys, f"{context.partition_key}.json")
        return os.path.join(self.base_path, *keys, "data.json")

    def handle_output(self, context: OutputContext, obj: Any):
        path = self._get_path(context)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        context.log.info(f"Saved output to {path}")

    def load_input(self, context: InputContext) -> Any:
        path = self._get_path(context)
        if not os.path.exists(path):
            # Fall back to unpartitioned path if the upstream asset is unpartitioned
            # but the downstream consumer is partitioned.
            keys = context.asset_key.path
            fallback = os.path.join(self.base_path, *keys, "data.json")
            if fallback != path and os.path.exists(fallback):
                path = fallback
            else:
                return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
