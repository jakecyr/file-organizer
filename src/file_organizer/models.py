from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import ollama

VISION_PROMPT = (
    "What is in this image? Describe the subject, visible text, screenshot or document "
    "type, and likely file category."
)


class OllamaModels:
    def __init__(self, *, host: str, embed_model: str, vision_model: str, naming_model: str | None):
        self.client = ollama.Client(host=host)
        self.embed_model = embed_model
        self.vision_model = vision_model
        self.naming_model = naming_model

    def describe_image(self, path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content = self._describe_image_with_prompt(encoded, VISION_PROMPT)
        if not content:
            content = self._describe_image_with_prompt(encoded, "What is shown in this image?")
        return " ".join(content.split())

    def _describe_image_with_prompt(self, encoded_image: str, prompt: str) -> str:
        response = self.client.chat(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image],
                }
            ],
            options={"temperature": 0},
        )
        message = _get(response, "message", {})
        return str(_get(message, "content", "")).strip()

    def embed(self, text: str) -> np.ndarray:
        try:
            response = self.client.embed(model=self.embed_model, input=text)
            embeddings = _get(response, "embeddings", None)
            if embeddings is not None:
                vector = (
                    embeddings[0] if embeddings and isinstance(embeddings[0], list) else embeddings
                )
                return _normalized(vector)
        except (AttributeError, TypeError):
            pass

        response = self.client.embeddings(model=self.embed_model, prompt=text)
        return _normalized(_get(response, "embedding", []))

    def name_cluster(self, summaries: list[str]) -> str | None:
        if not self.naming_model:
            return None
        content = "\n\n".join(summaries[:8])
        prompt = (
            "Name this file folder in 2 to 4 title-case words. "
            "Return only the folder name, no punctuation beyond spaces or hyphens.\n\n"
            f"{content}"
        )
        response = self.client.chat(
            model=self.naming_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        name = str(_get(_get(response, "message", {}), "content", "")).strip()
        return name or None


def _normalized(values: Any) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
