"""MiniMax music generation client.

AudioLDM2 ships a text-to-audio/music generation API that only loads local
AudioLDM checkpoints and runs the local latent-diffusion pipeline. This module
adds the MiniMax music-generation HTTP endpoint so the same text-to-music entry
point can also delegate generation to the MiniMax hosted API.

The endpoint configuration, request fields, output formats, audio formats and
response parsing all follow the MiniMax music-generation reference:

    POST /v1/music_generation  (global: api.minimax.io, CN: api.minimaxi.com)

Only standard-library modules and ``requests`` are used so the client stays
importable without the heavy AudioLDM2 torch stack.
"""

import requests

# ---------------------------------------------------------------------------
# Region configuration. Both the global (English) and the China endpoints are
# supported; the caller selects one with ``region="global_en"`` or
# ``region="cn_zh"``. URLs come from the MiniMax music-generation reference and
# must not be hardcoded elsewhere.
# ---------------------------------------------------------------------------
ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/music_generation",
    "cn_zh": "https://api.minimaxi.com/v1/music_generation",
}

# Current MiniMax music-generation models. The CN region additionally accepts
# the ``aigc_watermark`` regional field.
DEFAULT_MODEL = "music-3.0"
GENERATION_MODELS = ["music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free"]
COVER_MODELS = ["music-cover", "music-cover-free"]

# Fields accepted by the music-generation request, taken from the MiniMax
# music-generation reference. ``model`` is the only required field.
REQUEST_FIELDS = [
    "model",
    "prompt",
    "lyrics",
    "stream",
    "output_format",
    "audio_setting",
    "lyrics_optimizer",
    "is_instrumental",
    "audio_url",
    "audio_base64",
    "cover_feature_id",
]
REQUIRED_FIELDS = ["model"]

# Output formats. When ``stream`` is true only ``hex`` is allowed.
OUTPUT_FORMATS = ["url", "hex"]
STREAM_OUTPUT_FORMATS = ["hex"]
AUDIO_FORMATS = ["mp3", "wav", "pcm"]

# Regional fields. The China endpoint requires the AIGC watermark flag.
REGIONAL_FIELDS = {
    "global_en": [],
    "cn_zh": ["aigc_watermark"],
}

# Cover feature: one of audio_url / audio_base64 must be supplied, between 6 and
# 360 seconds and at most 50 MB.
COVER_INPUT_ONE_OF = ["audio_url", "audio_base64"]
COVER_INPUT_MIN_SECONDS = 6
COVER_INPUT_MAX_SECONDS = 360
COVER_INPUT_MAX_MB = 50

# Response parsing. ``data.status`` is 1 while generating and 2 once the audio
# is ready; ``data.audio`` carries the result and ``base_resp.status_code`` is 0
# on success. Music generation is synchronous, so there is no task id or query
# endpoint.
RESPONSE_STATUS_FIELD = "data.status"
RESPONSE_STATUS_VALUES = {"in_progress": 1, "completed": 2}
RESPONSE_AUDIO_FIELD = "data.audio"
RESPONSE_SUCCESS_CODE_FIELD = "base_resp.status_code"
RESPONSE_SUCCESS_CODE = 0


class MiniMaxMusicError(Exception):
    """Raised when the MiniMax music-generation API call fails."""


def _resolve_region(region):
    if region is None:
        region = "global_en"
    if region not in ENDPOINTS:
        raise ValueError(
            "Unsupported region '%s'. Expected one of: %s"
            % (region, ", ".join(sorted(ENDPOINTS.keys())))
        )
    return region


def _resolve_model(model, cover=False):
    if model is None:
        return COVER_MODELS[0] if cover else DEFAULT_MODEL
    allowed = COVER_MODELS if cover else GENERATION_MODELS
    if model not in allowed:
        raise ValueError(
            "Model '%s' is not a supported %s model. Expected one of: %s"
            % (model, "cover" if cover else "generation", ", ".join(allowed))
        )
    return model


def _build_headers(api_key):
    if not api_key:
        raise ValueError("A MiniMax API key (Bearer token) is required.")
    return {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }


def _apply_regional_fields(payload, region):
    """Inject region-specific fields required by the MiniMax endpoint."""
    for field in REGIONAL_FIELDS.get(region, []):
        # ``aigc_watermark`` defaults to true on the CN endpoint per the docs.
        payload.setdefault(field, True)
    return payload


def _validate_cover_input(payload):
    """Validate the cover-feature input (audio_url or audio_base64)."""
    provided = [f for f in COVER_INPUT_ONE_OF if payload.get(f)]
    if len(provided) != 1:
        raise ValueError(
            "Cover generation requires exactly one of: %s"
            % ", ".join(COVER_INPUT_ONE_OF)
        )


def _parse_response(response_json):
    """Parse the MiniMax music-generation response.

    Returns a dict with ``status``, ``audio`` and ``completed``. Raises
    :class:`MiniMaxMusicError` when the API reports a non-zero status code.
    """
    base_resp = response_json.get("base_resp", {}) or {}
    status_code = base_resp.get("status_code")
    if status_code is not None and int(status_code) != RESPONSE_SUCCESS_CODE:
        message = base_resp.get("status_msg") or "MiniMax API error"
        raise MiniMaxMusicError(
            "MiniMax music generation failed (status_code=%s): %s"
            % (status_code, message)
        )

    data = response_json.get("data", {}) or {}
    status = data.get("status")
    audio = data.get("audio")
    return {
        "status": status,
        "audio": audio,
        "completed": status == RESPONSE_STATUS_VALUES["completed"],
        "raw": response_json,
    }


def generate_music(
    api_key,
    model=None,
    prompt=None,
    lyrics=None,
    stream=False,
    output_format="url",
    audio_setting=None,
    lyrics_optimizer=None,
    is_instrumental=None,
    audio_url=None,
    audio_base64=None,
    cover_feature_id=None,
    region="global_en",
    cover=False,
    timeout=120,
):
    """Call the MiniMax music-generation endpoint.

    Parameters mirror the MiniMax music-generation reference request fields.
    ``model`` defaults to the current default music model (or the default cover
    model when ``cover`` is true) and is validated against the supported list
    rather than being hardcoded by callers.

    Returns the parsed response dict from :func:`_parse_response`.
    """
    region = _resolve_region(region)
    model = _resolve_model(model, cover=cover)

    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            "output_format '%s' is not supported. Expected one of: %s"
            % (output_format, ", ".join(OUTPUT_FORMATS))
        )
    if stream and output_format not in STREAM_OUTPUT_FORMATS:
        raise ValueError(
            "When stream is true output_format must be one of: %s"
            % ", ".join(STREAM_OUTPUT_FORMATS)
        )

    payload = {"model": model}
    optional = {
        "prompt": prompt,
        "lyrics": lyrics,
        "stream": stream,
        "output_format": output_format,
        "audio_setting": audio_setting,
        "lyrics_optimizer": lyrics_optimizer,
        "is_instrumental": is_instrumental,
        "audio_url": audio_url,
        "audio_base64": audio_base64,
        "cover_feature_id": cover_feature_id,
    }
    for key, value in optional.items():
        if value is not None:
            payload[key] = value

    if cover:
        _validate_cover_input(payload)
        if cover_feature_id is not None:
            payload["cover_feature_id"] = cover_feature_id

    _apply_regional_fields(payload, region)

    url = ENDPOINTS[region]
    headers = _build_headers(api_key)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        response_json = response.json()
    except ValueError:
        raise MiniMaxMusicError(
            "MiniMax returned a non-JSON response (HTTP %s): %s"
            % (response.status_code, response.text[:200])
        )
    if response.status_code >= 400:
        raise MiniMaxMusicError(
            "MiniMax music generation HTTP %s: %s"
            % (response.status_code, response_json)
        )
    return _parse_response(response_json)


def is_music_model(model_name):
    """Return True when ``model_name`` is a supported MiniMax music model."""
    return model_name in (GENERATION_MODELS + COVER_MODELS)
