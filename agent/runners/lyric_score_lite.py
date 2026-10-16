"""ACE-Step 1.5's lyric alignment score for the generated take alone.

ACE-Step's own get_lyric_score (acestep/core/generation/handler/
lyric_score.py, pinned commit 6d467e4b) runs the decoder on a BATCH OF TWO:
pure noise (the "lm" baseline) and the take's latent (the "dit" input). It
then stacks every captured layer's attention for both. Pulsar only uses
dit_score. On an RTX 3090 with a 228 s source (~5723 latent frames) the
batch of two asked for one more 8.25 GiB block on top of ~10 GiB it had
already taken, beside the ~12.3 GiB of loaded models, and ran out of memory
(run 6, 2026-10-16).

dit_lyric_score mirrors get_lyric_score step by step with batch size 1 and
scores the DiT half only. Every helper is ACE-Step's own, called on the
handler, so the number is the dit_score ACE-Step would give:
_resolve_custom_layers_config, _move_alignment_inputs_to_runtime,
_sample_noise_like, _load_model_context, _extract_lyric_segment,
_calculate_single_lyric_score and _lyric_score_error. The noise for the DiT
input is sampled from the same seed over the same shape as upstream, so
xt_dit is the same tensor.

Two things differ from upstream, neither in the result:
- The lyric rows of each layer are sliced out BEFORE the layers are
  stacked, so the stack copies only those rows, and the decoder's full
  attention outputs are dropped as soon as they are sliced.
- window_frames scores only the first N latent frames. The lyric token
  range comes from the text side alone (_extract_lyric_segment reads only
  the token ids, the language and the tokenizer), so it is unchanged.

Exceptions - a CUDA out-of-memory above all - propagate to the caller,
which decides whether to retry on a window.
"""

# ACE-Step's latents run at 25 frames a second: 48 kHz audio, 1920 samples
# per latent frame (conditioning_masks.py, conditioning_target.py).
LATENT_FRAMES_PER_SECOND = 48000 // 1920
# The fallback when the whole take does not fit: ~100 s from the start.
SCORE_WINDOW_FRAMES = 100 * LATENT_FRAMES_PER_SECOND


def latent_frames(tensor) -> int | None:
    """Frames in a [B, T, D] latent, or None if it has no shape."""
    shape = getattr(tensor, "shape", None)
    try:
        return int(shape[1])
    except (TypeError, IndexError, ValueError):
        return None


def dit_lyric_score(
    handler,
    pred_latent,
    encoder_hidden_states,
    encoder_attention_mask,
    context_latents,
    lyric_token_ids,
    vocal_language: str = "en",
    inference_steps: int = 8,
    seed: int = 42,
    window_frames: int | None = None,
) -> dict:
    """{"dit_score", "success", "error", "frames"} for one take. Raises on
    a runtime failure such as a CUDA out-of-memory."""
    import torch

    if handler.model is None:
        return _with_frames(handler._lyric_score_error("Model not initialized"), None)
    if not isinstance(inference_steps, int) or inference_steps <= 0:
        raise ValueError(f"inference_steps must be a positive non-zero integer, got {inference_steps!r}")

    custom_layers_config = handler._resolve_custom_layers_config(None)
    # Text side only: needs neither the decoder nor the latent.
    _, pure_lyric_ids, start_idx, end_idx = handler._extract_lyric_segment(
        lyric_token_ids=lyric_token_ids, vocal_language=vocal_language)

    with torch.inference_mode():
        pred_latent, encoder_hidden_states, encoder_attention_mask, context_latents = (
            handler._move_alignment_inputs_to_runtime(
                pred_latent=pred_latent,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            ))
        if window_frames is not None:
            pred_latent = pred_latent[:, :window_frames]
            context_latents = context_latents[:, :window_frames]
        frames = latent_frames(pred_latent)

        x0 = handler._sample_noise_like(pred_latent, seed)
        t_last_val = 1.0 / inference_steps
        xt_dit = t_last_val * x0 + (1.0 - t_last_val) * pred_latent
        del x0, pred_latent
        t_in = torch.tensor([t_last_val], device=xt_dit.device, dtype=xt_dit.dtype)
        attention_mask_in = torch.ones(1, xt_dit.shape[1], device=xt_dit.device, dtype=xt_dit.dtype)

        with handler._load_model_context("model"):
            decoder = handler.model.decoder
            if hasattr(decoder, "eval"):
                decoder.eval()
            decoder_outputs = decoder(
                hidden_states=xt_dit,
                timestep=t_in,
                timestep_r=t_in,
                attention_mask=attention_mask_in,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                past_key_values=None,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                output_attentions=True,
                custom_layers_config=custom_layers_config,
                enable_early_exit=True,
            )
        del xt_dit, encoder_hidden_states, encoder_attention_mask, context_latents, attention_mask_in

        attentions = decoder_outputs[2]
        del decoder_outputs
        if attentions is None:
            return _with_frames(handler._lyric_score_error("Model did not return attentions"), frames)

        lyric_rows = []
        for layer_attn in attentions:
            if layer_attn is None:
                continue
            # [1, heads, T_latent, T_text] -> [1, heads, T_text, T_latent]
            text_major = layer_attn.transpose(-1, -2)
            if start_idx >= text_major.shape[-2]:
                return _with_frames(handler._lyric_score_error("Lyrics indices out of bounds"), frames)
            lyric_rows.append(text_major[..., start_idx:end_idx, :])
        del attentions
        if not lyric_rows:
            return _with_frames(handler._lyric_score_error("No valid attention layers returned"), frames)

        # [layers, 1, heads, lyric, T_latent] -> [layers, heads, lyric, T_latent],
        # as upstream squeezes its bsz == 1 halves.
        pure_matrix_dit = torch.stack(lyric_rows).squeeze(1)
        del lyric_rows

        from acestep.core.scoring.dit_score import MusicLyricScorer

        aligner = MusicLyricScorer(handler.text_tokenizer)
        dit_score = handler._calculate_single_lyric_score(
            aligner=aligner,
            matrix=pure_matrix_dit,
            pure_lyric_ids=pure_lyric_ids,
            custom_layers_config=custom_layers_config,
        )
    return {"dit_score": dit_score, "success": True, "error": None, "frames": frames}


def _with_frames(outcome: dict, frames: int | None) -> dict:
    return {**outcome, "frames": frames}
