from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TOP_DOMAIN_SLUGS = {
    "object",
    "action",
    "dialogue",
    "mind",
    "person",
    "place",
    "emotion",
    "language",
    "number",
    "nature",
    "perception",
    "body",
}

LEGACY_TOPIC_SLUGS = {
    "superai",
    "graph",
    "memory",
    "checkpoint",
    "semantic_vector",
    "feedback",
    "learning",
}

SLUG_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)


@dataclass(frozen=True)
class CanonicalResolution:
    uri: str
    canonical: str
    alias: str | None = None
    redirected: bool = False


class CanonicalResolver:
    def __init__(self, checkpoint: Any | None = None) -> None:
        self.checkpoint = checkpoint

    def resolve(self, uri: str) -> CanonicalResolution:
        return _resolve_uri(uri, self.checkpoint)

    def canonical_uri(self, uri: str) -> str:
        return self.resolve(uri).canonical

    def register_surface(self, canonical_uri: str, surface: str, lang: str | None = None) -> None:
        if not self.checkpoint or not canonical_uri or not surface:
            return
        canonical = self.canonical_uri(canonical_uri)
        surface = _clean_surface(surface)
        if not surface:
            return
        surface_forms = _ensure_mapping(self.checkpoint, "surface_forms")
        bucket = surface_forms.setdefault(canonical, {})
        if not isinstance(bucket, dict):
            bucket = {}
            surface_forms[canonical] = bucket
        if lang in {"ru", "en"}:
            values = bucket.setdefault(lang, [])
            if isinstance(values, list) and surface not in values:
                values.append(surface)
        aliases = _ensure_mapping(self.checkpoint, "aliases")
        if surface not in aliases:
            aliases[surface] = canonical

    def register_concept(self, canonical_uri: str, *, label: str | None = None, aliases: list[str] | None = None, lang: str | None = None, source_uri: str | None = None, quality: float | None = None) -> str:
        if not self.checkpoint:
            return self.canonical_uri(canonical_uri)
        canonical = self.canonical_uri(canonical_uri)
        concepts = _ensure_mapping(self.checkpoint, "canonical_concepts")
        item = concepts.setdefault(canonical, {})
        if not isinstance(item, dict):
            item = {}
            concepts[canonical] = item
        item.setdefault("uri", canonical)
        if label:
            item["label"] = _clean_surface(label)
        if lang in {"ru", "en"}:
            langs = item.setdefault("langs", [])
            if isinstance(langs, list) and lang not in langs:
                langs.append(lang)
        if source_uri:
            sources = item.setdefault("source_uris", [])
            if isinstance(sources, list) and source_uri not in sources:
                sources.append(source_uri)
        if quality is not None:
            item["quality"] = max(float(item.get("quality", 0.0) or 0.0), float(quality))
        if aliases:
            existing = item.setdefault("aliases", [])
            if isinstance(existing, list):
                for alias in aliases:
                    clean = _clean_surface(alias)
                    if clean and clean not in existing:
                        existing.append(clean)
                        self.register_surface(canonical, clean, lang=lang)
        if label:
            self.register_surface(canonical, label, lang=lang)
        redirects = _ensure_mapping(self.checkpoint, "concept_redirects")
        redirects[canonical] = canonical
        return canonical


def canonical_concept_uri(text: str, lang: str | None = None) -> str:
    slug = canonical_slug(text)
    if not slug:
        return ""
    if lang in {"ru", "en"}:
        return f"/m/concept/{slug}"
    return f"/m/concept/{slug}"


def canonical_slug(text: str) -> str:
    clean = _clean_surface(text).casefold().replace(" ", "_")
    clean = SLUG_RE.sub("_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def _resolve_uri(uri: str, checkpoint: Any | None = None) -> CanonicalResolution:
    clean = _clean_uri(uri)
    if not clean:
        return CanonicalResolution(uri="", canonical="")
    if checkpoint is not None:
        redirects = getattr(checkpoint, "concept_redirects", {})
        if isinstance(redirects, dict):
            redirect = str(redirects.get(clean, ""))
            if redirect:
                return CanonicalResolution(uri=clean, canonical=redirect, redirected=redirect != clean)
    if clean.startswith("/m/top/"):
        slug = clean.rsplit("/", 1)[-1]
        if slug in TOP_DOMAIN_SLUGS:
            return CanonicalResolution(uri=clean, canonical=clean)
        if slug in LEGACY_TOPIC_SLUGS:
            return CanonicalResolution(uri=clean, canonical=f"/m/concept/{slug}", redirected=True)
        return CanonicalResolution(uri=clean, canonical=clean)
    if clean.startswith("/c/"):
        parts = clean.split("/", 3)
        if len(parts) > 3:
            slug = canonical_slug(parts[3])
            return CanonicalResolution(uri=clean, canonical=f"/m/concept/{slug}", alias=parts[3], redirected=True)
    if clean.startswith("/m/concept/"):
        return CanonicalResolution(uri=clean, canonical=clean)
    if clean.startswith("/m/"):
        slug = clean.rsplit("/", 1)[-1]
        if slug in TOP_DOMAIN_SLUGS:
            return CanonicalResolution(uri=clean, canonical=f"/m/top/{slug}", redirected=clean != f"/m/top/{slug}")
        if slug in LEGACY_TOPIC_SLUGS and clean.count("/") <= 2:
            return CanonicalResolution(uri=clean, canonical=f"/m/concept/{canonical_slug(slug)}", redirected=True)
        return CanonicalResolution(uri=clean, canonical=clean)
    return CanonicalResolution(uri=clean, canonical=clean)


def _clean_surface(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split()).strip()


def _clean_uri(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _looks_like_surface_slug(value: str) -> bool:
    return bool(value) and value not in TOP_DOMAIN_SLUGS and not value.startswith("top/")


def _ensure_mapping(checkpoint: Any, key: str) -> dict[str, Any]:
    raw = getattr(checkpoint, key, None)
    if isinstance(raw, dict):
        return raw
    mapping: dict[str, Any] = {}
    setattr(checkpoint, key, mapping)
    return mapping
