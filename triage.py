"""LLM-backed customer support triage logic."""

from __future__ import annotations

import importlib
import json
import logging
import os
import contextvars
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from prompt import get_system_prompt
import validator


logger = logging.getLogger(__name__)
BATCH_CONTEXT: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "triage_batch_number",
    default=None,
)
BATCH_METRICS_CONTEXT: contextvars.ContextVar[Dict[str, int]] = contextvars.ContextVar(
    "triage_batch_metrics",
    default={"api_calls_made": 0, "retry_count": 0, "quota_exceeded": 0},
)


class TriageError(Exception):
    """Base exception for triage failures."""


class ConfigurationError(TriageError):
    """Raised when the LLM configuration is invalid or incomplete."""


class ProviderError(TriageError):
    """Raised when the configured LLM provider cannot be called successfully."""


class ResponseParseError(TriageError):
    """Raised when the LLM response cannot be parsed as valid JSON."""


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for the selected LLM provider."""

    provider: str
    model: str
    api_key: str


def analyze_message(message: str) -> Dict[str, Any]:
    """Analyze a customer message and return the provider's JSON response.

    The function combines the system prompt with the customer message, sends the
    request to the configured LLM provider, and returns the parsed JSON object.
    """

    results = analyze_messages([message])
    return results[0]


def analyze_messages(messages: list[str]) -> list[Dict[str, Any]]:
    """Analyze a batch of customer messages and return validated results.

    Each batch is sent as one LLM request, then normalized into one result per
    input message while preserving order.
    """

    if not isinstance(messages, list):
        raise TypeError("messages must be a list of strings")

    normalized_messages = [_normalize_message(message) for message in messages]
    if not normalized_messages:
        return []

    reset_batch_metrics()
    config = _load_config()
    logger.info(
        "triage batch request received",
        extra={
            "event": "triage_batch_request_received",
            "provider": config.provider,
            "model": config.model,
            "batch_number": BATCH_CONTEXT.get(),
            "message_count": len(normalized_messages),
        },
    )

    payload = _build_request_payload(config.provider, normalized_messages)
    raw_response = _call_provider_with_backoff(config, payload, len(normalized_messages))
    parsed_responses = _parse_json_array_response(raw_response)
    results = _normalize_batch_responses(parsed_responses, len(normalized_messages))

    logger.info(
        "triage batch request completed",
        extra={
            "event": "triage_batch_request_completed",
            "provider": config.provider,
            "batch_number": BATCH_CONTEXT.get(),
            "message_count": len(normalized_messages),
            "result_count": len(results),
        },
    )
    return results


def reset_batch_metrics() -> None:
    """Reset per-batch API metrics for diagnostics."""

    BATCH_METRICS_CONTEXT.set({"api_calls_made": 0, "retry_count": 0, "quota_exceeded": 0})


def get_batch_metrics() -> Dict[str, int]:
    """Return the current per-batch API metrics."""

    return dict(BATCH_METRICS_CONTEXT.get())


def set_batch_context(batch_number: Optional[int]) -> None:
    """Set the current batch number for structured logging."""

    BATCH_CONTEXT.set(batch_number)


def _normalize_message(message: str) -> str:
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    stripped = message.strip()
    if not stripped:
        raise ValueError("message must not be empty")

    return stripped


def _load_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "gemini"}:
        raise ConfigurationError(
            "LLM_PROVIDER must be set to 'openai' or 'gemini'"
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    else:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    if not api_key:
        raise ConfigurationError(f"Missing API key for provider '{provider}'")
    if not model:
        raise ConfigurationError(f"Missing model name for provider '{provider}'")

    return LLMConfig(provider=provider, model=model, api_key=api_key)


def _build_request_payload(provider: str, messages: list[str]) -> Mapping[str, Any]:
    system_prompt = get_system_prompt()
    user_content = _build_batch_user_content(messages)

    if provider == "openai":
        return {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
        }

    return {
        "system_prompt": system_prompt,
        "message": user_content,
        "generation_config": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }


def _build_batch_user_content(messages: list[str]) -> str:
    lines = [
        "Analyze the following customer messages and return a JSON array with one object per message in the same order.",
        "",
        "Messages:",
    ]
    for index, message in enumerate(messages, start=1):
        lines.append(f"{index}. {message}")
    return "\n".join(lines)


def _call_provider(config: LLMConfig, payload: Mapping[str, Any]) -> str:
    if config.provider == "openai":
        return _call_openai(config, payload)
    if config.provider == "gemini":
        return _call_gemini(config, payload)
    raise ConfigurationError(f"Unsupported provider: {config.provider}")


def _call_provider_with_backoff(
    config: LLMConfig, payload: Mapping[str, Any], message_count: int
) -> str:
    for attempt in range(2):
        try:
            _increment_batch_metric("api_calls_made")
            logger.info(
                "API request started",
                extra={
                    "event": "api_request_started",
                    "batch_number": BATCH_CONTEXT.get(),
                    "message_count": message_count,
                    "attempt": attempt + 1,
                },
            )
            raw_response = _call_provider(config, payload)
            logger.info(
                "API request completed",
                extra={
                    "event": "api_request_completed",
                    "batch_number": BATCH_CONTEXT.get(),
                    "message_count": message_count,
                    "attempt": attempt + 1,
                },
            )
            return raw_response
        except ProviderError as exc:
            if _is_retryable_temporary_error(exc) and attempt == 0:
                _increment_batch_metric("retry_count")
                delay_seconds = 2
                batch_number = BATCH_CONTEXT.get()
                retry_message = f"Retrying batch {batch_number} after temporary error" if batch_number is not None else "Retrying batch after temporary error"
                logger.warning(
                    retry_message,
                    extra={
                        "event": "triage_batch_retry",
                        "batch_number": batch_number,
                        "message_count": message_count,
                        "attempt": 1,
                        "delay_seconds": delay_seconds,
                        "error": str(exc),
                    },
                )
                time.sleep(delay_seconds)
                continue

            if config.provider == "gemini" and _is_http_429(exc):
                _increment_batch_metric("quota_exceeded")
                logger.info("Gemini quota exceeded.")
                logger.warning("Falling back to safe response.")
                return _build_gemini_quota_fallback(message_count)

            raise


def _is_retryable_temporary_error(error: Exception) -> bool:
    candidates = [error, getattr(error, "__cause__", None), getattr(error, "__context__", None)]
    for candidate in candidates:
        if candidate is None:
            continue

        status_code = getattr(candidate, "status_code", None)
        if status_code in {429, 503}:
            return True

        code = getattr(candidate, "code", None)
        if code in {429, 503}:
            return True

        response = getattr(candidate, "response", None)
        if getattr(response, "status_code", None) in {429, 503}:
            return True

        text = str(candidate).lower()
        name = candidate.__class__.__name__.lower()
        if (
            "429" in text
            or "503" in text
            or "resourceexhausted" in name
            or "too many requests" in text
            or "timeout" in text
            or "timed out" in text
            or "connection" in text and "timeout" in text
        ):
            return True

    return False


def _call_openai(config: LLMConfig, payload: Mapping[str, Any]) -> str:
    try:
        openai_module = importlib.import_module("openai")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigurationError(
            "The openai package is required when LLM_PROVIDER=openai"
        ) from exc

    try:
        client = openai_module.OpenAI(api_key=config.api_key)
        response = client.chat.completions.create(**payload)
    except Exception as exc:
        logger.error(
            "triage provider error",
            extra={
                "event": "triage_provider_error",
                "provider": config.provider,
                "error": str(exc),
            },
        )
        raise ProviderError(f"OpenAI request failed: {exc}") from exc

    content = _extract_openai_content(response)
    if content is None:
        raise ProviderError("OpenAI response did not include message content")
    return content


def _extract_openai_content(response: Any) -> Optional[str]:
    try:
        choices = response.choices
        if not choices:
            return None
        message = choices[0].message
        return message.content
    except Exception:
        return None


def _call_gemini(config: LLMConfig, payload: Mapping[str, Any]) -> str:
    try:
        genai = importlib.import_module("google.generativeai")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigurationError(
            "The google-generativeai package is required when LLM_PROVIDER=gemini"
        ) from exc

    try:
        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(
            config.model,
            system_instruction=payload["system_prompt"],
        )
        response = model.generate_content(
            payload["message"],
            generation_config=payload["generation_config"],
        )
    except Exception as exc:
        logger.exception("Gemini provider error")
        raise ProviderError(f"Gemini request failed: {exc}") from exc

    text = _extract_gemini_text(response)
    if text is None:
        raise ProviderError("Gemini response did not include text content")
    return text


def _is_http_429(error: Exception) -> bool:
    candidates = [error, getattr(error, "__cause__", None), getattr(error, "__context__", None)]
    for candidate in candidates:
        if candidate is None:
            continue

        status_code = getattr(candidate, "status_code", None)
        if status_code == 429:
            return True

        code = getattr(candidate, "code", None)
        if code == 429:
            return True

        response = getattr(candidate, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True

        text = str(candidate).lower()
        name = candidate.__class__.__name__.lower()
        if "429" in text or "resourceexhausted" in name or "quota" in text or "too many requests" in text:
            return True

    return False


def _build_gemini_quota_fallback(batch_size: int) -> str:
    fallback_item = {
        "category": "Unknown",
        "priority": "P3",
        "summary": "LLM unavailable due to API quota.",
        "suggested_action": "Retry later or route to human support.",
        "needs_human": True,
        "confidence": 0.0,
        "processing_status": "Quota Exceeded",
    }
    return json.dumps([fallback_item for _ in range(max(batch_size, 1))])


def _extract_gemini_text(response: Any) -> Optional[str]:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        parts = candidates[0].content.parts
        text_parts = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                text_parts.append(part_text)
        combined = "".join(text_parts).strip()
        return combined or None
    except Exception:
        return None


def _parse_json_response(raw_response: str) -> Dict[str, Any]:
    if not isinstance(raw_response, str):
        raise ResponseParseError("LLM response was not text")

    content = raw_response.strip()
    if not content:
        raise ResponseParseError("LLM response was empty")

    content = _strip_json_fences(content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ResponseParseError(f"LLM response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ResponseParseError("LLM response JSON must be an object")

    return parsed


def _parse_json_array_response(raw_response: str) -> list[Dict[str, Any]]:
    if not isinstance(raw_response, str):
        raise ResponseParseError("LLM response was not text")

    content = raw_response.strip()
    if not content:
        raise ResponseParseError("LLM response was empty")

    content = _strip_json_fences(content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ResponseParseError(f"LLM response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise ResponseParseError("LLM response JSON must be an array")

    return parsed


def _normalize_batch_responses(
    responses: list[Any], expected_count: int
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    if len(responses) != expected_count:
        logger.warning(
            "triage batch response length mismatch",
            extra={
                "event": "triage_batch_response_length_mismatch",
                "expected_count": expected_count,
                "actual_count": len(responses),
            },
        )

    for index in range(expected_count):
        item = responses[index] if index < len(responses) else None
        results.append(_normalize_batch_item(item))

    return results


def _normalize_batch_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return _default_validated_response()

    try:
        return validator.validate_response(item)
    except Exception:
        return _default_validated_response()


def _default_validated_response() -> Dict[str, Any]:
    return validator.validate_response({})


def _strip_json_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        if content.endswith("```"):
            return content[3:-3].strip()
    return content


def _increment_batch_metric(metric_name: str) -> None:
    metrics = dict(BATCH_METRICS_CONTEXT.get())
    metrics[metric_name] = metrics.get(metric_name, 0) + 1
    BATCH_METRICS_CONTEXT.set(metrics)


def _apply_confidence_guardrail(response: Dict[str, Any]) -> None:
    # Low-confidence predictions must route to a human.
    confidence = response.get("confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.70:
        response["needs_human"] = True
