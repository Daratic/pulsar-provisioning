> **Superseded 2026-07-30** — this documents the abandoned prebuilt-image approach. ACE-Step now deploys as a BYOM runner via `agent/runners/ace_step.py`; see `docs/superpowers/specs/2026-07-30-pulsar-ace-step-cover-byom-design.md` for the current approach.

# ACE-Step API Findings

This document contains the research findings regarding the official Vast.ai ACE-Step template's bundled REST API.

## Executive Summary & Unconfirmed Items

*   **Server Identity:** The server bundled in the `vastai/acestep` template is **NOT** the ComfyUI server referenced in Task 1. It is a bespoke FastAPI application (`acestep/api_server.py`) wrapping the inference code directly.
*   **Vast.ai Template README:** The link `https://cloud.vast.ai/template/readme/bf1389ff71ee2d4008411a30ec21ca19` could not be fetched by the agent because it redirects to the Vast.ai console login page (React SPA). Findings below rely on the public Docker Hub image (`vastai/acestep`) and its upstream GitHub repo (`ace-step/ACE-Step-1.5`).
*   **UNCONFIRMED - `OPEN_BUTTON_TOKEN`:** The source code does not contain any reference to `OPEN_BUTTON_TOKEN`. The server natively authenticates via an `ACESTEP_API_KEY` environment variable. It is highly likely the Vast.ai template maps `OPEN_BUTTON_TOKEN` to `ACESTEP_API_KEY` at runtime, but this cannot be confirmed from public source.
*   **UNCONFIRMED - External Port (18001):** The internal port is confirmed as 8001, but the external port mapping (18001) is managed by Vast.ai.
*   **Recommendation for Controller/Tom:** Because `OPEN_BUTTON_TOKEN` mapping and exact Vast.ai port behavior cannot be confirmed from the public GitHub source alone, it is recommended to rent a real instance (~$0.003/minute) for 5-10 minutes to explicitly verify the `/docs` endpoint, the `OPEN_BUTTON_TOKEN` behavior, and the port mappings. This requires a human decision to spend funds.

## Detailed Findings

### (a) Submit Cover Job API

*   **Endpoint:** `POST /release_task` (Source: `acestep/api/http/release_task_route.py`, line 47)
*   **Request Body Shape:** JSON or Multipart Form matching the `GenerateMusicRequest` Pydantic model (Source: `acestep/api/http/release_task_models.py`, lines 12-125).
*   **Key Fields for Cover Jobs:**
    *   `task_type`: String (default `"text2music"`, likely set to `"cover"` for style transfer).
    *   `audio_cover_strength`: Float (default `1.0`).
    *   `cover_noise_strength`: Float (default `0.0`, description: "Cover noise blending strength (0.0=pure noise, 1.0=closest to source audio). Used for cover/repaint tasks.").
    *   `src_audio_path`: Optional string (path to source audio).
    *   `reference_audio_path`: Optional string.
    *   `prompt`: String.
    *   `lyrics`: String.
    *   `bpm`: Optional integer.
    *   `key_scale`: String.
    *   `time_signature`: String.
    *   `vocal_language`: String (default `"en"`).
    *   `inference_steps`: Integer (default `8`).
    *   `guidance_scale`: Float (default `7.0`).
    *   `seed`: Integer or String (default `-1`).

### (b) Polling Job Completion

*   **Endpoint:** `POST /query_result` (Source: `acestep/api/http/query_result_route.py`, line 35)
*   **Request Body Shape:** `{"task_id_list": ["task1_id", "task2_id"]}` (Source: `acestep/api/http/query_result_service.py`, `parse_task_id_list` function, line 10)
*   **Response Shape:** A JSON array of objects.
    *   `task_id`: String
    *   `status`: Integer (Status code mapper)
    *   `progress_text`: String
    *   `result`: **A JSON-encoded string** that must be parsed by the client. Once parsed, it is an array of objects containing:
        *   `file`: String (Audio file URL, fetched via GET /v1/audio - see docs/en/API.md)
        *   `wave`: String
        *   `status`: Integer
        *   `create_time`: Integer
        *   `env`: String
        *   `prompt`: String
        *   `lyrics`: String
        *   `metas`: Object containing `bpm`, `duration`, `genres`, `keyscale`, `timesignature`
        *   *Source:* `_build_store_result_payload` in `acestep/api/http/query_result_service.py`, lines 84-101.

### (c) Authentication Mechanism

*   **Mechanism:** The server expects an API key.
*   **Implementation:** The API key is defined by the `ACESTEP_API_KEY` environment variable. The client can provide this key in two ways:
    1.  As an `ai_token` field in the request body.
    2.  As a Bearer token in the `Authorization` header (`Authorization: Bearer <token>`).
    *   *Source:* `verify_token_from_request` and `verify_api_key` functions in `acestep/api/http/auth.py`, lines 23-76.
*   **`OPEN_BUTTON_TOKEN`:** Unconfirmed. Not found in the source code.

### (d) Ports

*   **Internal Port:** 8001. Confirmed via `start_api_server.sh` (line 14) and `Dockerfile` (`EXPOSE 8001`, line 97).
*   **External Port (18001):** Unconfirmed.
*   **Implementation Note:** To connect dynamically, the application reads the `VAST_TCP_PORT_8001` environment variable over SSH inside the running instance. This environment variable is set by Vast.ai and contains the external port mapped to internal port 8001 (see docs.vast.ai/documentation/instances/connect/networking).
