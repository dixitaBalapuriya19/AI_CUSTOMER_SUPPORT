"""Validation and normalization helpers for triage responses."""

from __future__ import annotations

from typing import Any, Dict, Mapping


class ValidationError(ValueError):
    """Raised when the triage response cannot be normalized safely."""


_DEFAULTS = {
    "category": "Other",
    "priority": "P3",
    "summary": "No summary available.",
    "suggested_action": "Escalate for manual review.",
    "needs_human": True,
    "confidence": 0.0,
}

_ALLOWED_CATEGORIES = {
    "Billing",
    "Technical Support",
    "Account",
    "Order",
    "Complaint",
    "Feature Request",
    "General Inquiry",
    "Spam",
    "Other",
}

_ALLOWED_PRIORITIES = {"P0", "P1", "P2", "P3"}


def validate_response(data: dict) -> dict:
    """Return a cleaned triage response with safe defaults and guardrails.

    The function is intentionally defensive: it accepts an arbitrary dictionary,
    fills missing values, normalizes known fields, and returns a compact payload
    containing only the supported schema keys.
    """

    if not isinstance(data, dict):
        raise ValidationError("response must be a dictionary")

    cleaned: Dict[str, Any] = {}
    cleaned["category"] = _normalize_category(data.get("category"))
    cleaned["priority"] = _normalize_priority(data.get("priority"))
    cleaned["summary"] = _normalize_text(
        data.get("summary"), _DEFAULTS["summary"]
    )
    cleaned["suggested_action"] = _normalize_text(
        data.get("suggested_action"), _DEFAULTS["suggested_action"]
    )
    cleaned["confidence"] = _normalize_confidence(data.get("confidence"))
    cleaned["needs_human"] = _normalize_needs_human(
        data.get("needs_human"), cleaned["confidence"]
    )

    return cleaned


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return _DEFAULTS["category"]

    normalized = value.strip().lower()
    mapping = {
        "billing": "Billing",
        "technical support": "Technical Support",
        "technical": "Technical Support",
        "tech support": "Technical Support",
        "account": "Account",
        "order": "Order",
        "shipping": "Order",
        "refund": "Billing",
        "complaint": "Complaint",
        "feature request": "Feature Request",
        "feature": "Feature Request",
        "general inquiry": "General Inquiry",
        "question": "General Inquiry",
        "spam": "Spam",
        "other": "Other",
    }
    return mapping.get(normalized, _DEFAULTS["category"])


def _normalize_priority(value: Any) -> str:
    if not isinstance(value, str):
        return _DEFAULTS["priority"]

    normalized = value.strip().upper()
    if normalized in _ALLOWED_PRIORITIES:
        return normalized
    return _DEFAULTS["priority"]


def _normalize_text(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default

    normalized = value.strip()
    return normalized if normalized else default


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if confidence < 0.0:
        confidence = 0.0
    elif confidence > 1.0:
        confidence = 1.0

    return confidence


def _normalize_needs_human(value: Any, confidence: float) -> bool:
    if confidence < 0.70:
        return True
    if isinstance(value, bool):
        return value
    return _DEFAULTS["needs_human"]
