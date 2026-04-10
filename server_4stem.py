"""
─────────────────────────────────────────────────────────────────────────────
 FLASK SERVER  —  4-Stem Hybrid Architecture  (SR-fixed)

 Critical fix: model was trained at 16000 Hz.
 Audio is loaded at 44100, downsampled to 16000 for ML model,
 ML outputs upsampled back to 44100 for saving.
 DSP runs at 44100 directly (frame sizes scale with SR).

 Speech  → ML model  (44k → downsample to 16k → model → upsample to 44k)
 Music   → ML model  (same)
 Noise   → ML model  (same)
 Impacts → DSP transient detector  (44k, no ML involved)

 ML model always runs on original unmodified mixture.
 DSP runs separately and completely independently.
 Zero interaction between DSP and ML stems.
─────────────────────────────────────────────────────────────────────────────
"""

import os, uuid, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from torchaudio.models import HDemucs

app = Flask(__name__)
CORS(app)

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEMS      = ['Speech', 'Music', 'Impacts', 'Noise']
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
STEM_DIR   = os.path.join(os.path.dirname(__file__), "stems_out")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STEM_DIR,   exist_ok=True)

TARGET_SR = 44100   # output / save sample rate
MODEL_SR  = 16000   # model trained at this SR — DO NOT CHANGE

# ── Resampler cache (avoid rebuilding every request) ─────────────────────────
_resamplers = {}
def get_resampler(from_sr, to_sr):
    key = (from_sr, to_sr)
    if key not in _resamplers:
        _resamplers[key] = T.Resample(from_sr, to_sr)
    return _resamplers[key]

# ── Load model ───────────────────────────────────────────────────────────────
model = HDemucs(sources=STEMS, audio_channels=2)
model = nn.DataParallel(model)

CKPT_PATH = os.path.join(os.path.dirname(__file__), "80epochbst.pth")
if os.path.exists(CKPT_PATH):
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt.get('model', ckpt)
    model.module.load_state_dict(state, strict=False)
    epoch    = ckpt.get('epoch', '?')
    val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', '?'))
    print(f"✅ Model loaded  epoch={epoch}  best_val_loss={val_loss}")
else:
    print(f"⚠  Checkpoint not found at {CKPT_PATH} — running with random weights")

model = model.to(DEVICE).eval()
print(f"   Device    : {DEVICE}")
print(f"   Model SR  : {MODEL_SR} Hz  (trained at this rate)")
print(f"   Output SR : {TARGET_SR} Hz  (files saved at this rate)")


# =============================================================================
# DSP TRANSIENT DETECTOR
# Pure math — no ML, no weights, no training.
# Produces Impacts channel only.
# Never touches Speech / Music / Noise.
#
# Speech plosive protection (2 layers):
#   1. Min duration filter  — rejects bursts < 15ms
#      Plosives = 5-10ms,  real impacts = 15ms+
#   2. Spectral check       — rejects bursts where >75% energy above 2kHz
#      Plosives = sharp high-freq click
#      Impacts  = full spectrum including bass
# =============================================================================
def extract_impacts_dsp(mixture_wav, sr,
                        ratio_thresh  = 8.0,
                        energy_floor  = 1e-5,
                        min_dur_ms    = 15.0,
                        pre_roll_ms   = 30.0,
                        post_roll_ms  = 150.0):
    """
    mixture_wav : (2, T) at sample rate sr
    Returns: impacts_wav (2,T), impact_mask (T,), n_events int
    """
    device    = mixture_wav.device
    mono      = mixture_wav.mean(0)
    T_len     = mono.shape[0]

    frame_len = int(sr * 0.020)            # 20ms — scales with SR automatically
    hop       = int(sr * 0.005)            # 5ms hop

    if T_len < frame_len:
        return (torch.zeros_like(mixture_wav),
                torch.zeros(T_len, device=device),
                0)

    # ── 1. Short-time energy ──────────────────────────────────────────────
    frames   = mono.unfold(0, frame_len, hop)
    energy   = frames.pow(2).mean(-1)
    n_frames = energy.shape[0]

    # ── 2. Slow background envelope (500ms median) ────────────────────────
    med_win  = max(1, int(0.500 / 0.005))
    slow_env = torch.zeros_like(energy)
    for i in range(n_frames):
        s = max(0, i - med_win // 2)
        e = min(n_frames, i + med_win // 2)
        slow_env[i] = energy[s:e].median()

    # ── 3. Transient ratio ─────────────────────────────────────────────────
    ratio       = energy / (slow_env + 1e-8)
    raw_detects = (ratio > ratio_thresh) & (energy > energy_floor)

    # ── 4. SPEECH PROTECTION 1: minimum duration ─────────────────────────
    # Plosives (P,T,K) = 5-10ms → rejected below min_dur_ms=15ms threshold
    # Real impacts = 15ms+      → accepted
    min_dur_frames = max(1, int(min_dur_ms / 5.0))
    impact_frames  = torch.zeros(n_frames, dtype=torch.bool, device=device)
    i = 0
    while i < n_frames:
        if raw_detects[i]:
            j = i
            while j < n_frames and raw_detects[j]:
                j += 1
            if (j - i) >= min_dur_frames:
                impact_frames[i:j] = True
            i = j
        else:
            i += 1

    # ── 5. SPEECH PROTECTION 2: spectral centroid check ──────────────────
    # Plosives: >75% energy above 2kHz → reject as plosive
    # Impacts:  energy spread across full spectrum → keep
    HF_CUTOFF_HZ      = 2000.0
    PLOSIVE_HF_THRESH = 0.75
    hf_bin = max(1, int(HF_CUTOFF_HZ / (sr / frame_len)))
    i = 0
    while i < n_frames:
        if impact_frames[i]:
            j = i
            while j < n_frames and impact_frames[j]:
                j += 1
            s_samp = i * hop
            e_samp = min(T_len, j * hop + frame_len)
            burst  = mono[s_samp:e_samp]
            if burst.shape[0] >= frame_len:
                fft_mag = torch.abs(torch.fft.rfft(burst[:frame_len]))
                total   = fft_mag.pow(2).sum() + 1e-8
                hf      = fft_mag[hf_bin:].pow(2).sum()
                if (hf / total).item() > PLOSIVE_HF_THRESH:
                    impact_frames[i:j] = False   # mostly HF → plosive → reject
            i = j
        else:
            i += 1

    # ── 6. Count distinct impact events ──────────────────────────────────
    n_events, prev = 0, False
    for fi in range(n_frames):
        cur = bool(impact_frames[fi].item())
        if cur and not prev:
            n_events += 1
        prev = cur

    # ── 7. Expand with pre/post roll ──────────────────────────────────────
    pre_frames  = max(1, int(pre_roll_ms  / 5.0))
    post_frames = max(1, int(post_roll_ms / 5.0))
    expanded    = impact_frames.clone()
    for idx in impact_frames.nonzero(as_tuple=True)[0]:
        s = max(0, idx.item() - pre_frames)
        e = min(n_frames, idx.item() + post_frames)
        expanded[s:e] = True

    # ── 8. Map to sample domain ───────────────────────────────────────────
    mask_wav = torch.zeros(T_len, device=device)
    for fi in range(n_frames):
        if expanded[fi]:
            s = fi * hop
            e = min(T_len, s + frame_len)
            mask_wav[s:e] = 1.0

    # ── 9. Smooth edges (5ms fade) to prevent clicks ──────────────────────
    fade_len = max(1, int(sr * 0.005))
    kernel   = torch.ones(1, 1, fade_len * 2 + 1, device=device) / (fade_len * 2 + 1)
    mask_wav = F.conv1d(
        mask_wav.unsqueeze(0).unsqueeze(0),
        kernel, padding=fade_len
    ).squeeze().clamp(0.0, 1.0)

    impacts_wav = mixture_wav * mask_wav.unsqueeze(0)
    return impacts_wav, mask_wav, n_events


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stems/<filename>")
def serve_stem(filename):
    return send_from_directory(STEM_DIR, filename)

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/separate", methods=["POST"])
def separate():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file    = request.files["audio"]
    job_id  = uuid.uuid4().hex[:8]
    in_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    file.save(in_path)

    try:
        # ── Load & normalize to TARGET_SR stereo ──────────────────────────
        wav, sr = torchaudio.load(in_path)
        if sr != TARGET_SR:
            wav = get_resampler(sr, TARGET_SR)(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        # wav : (2, T) at 44100

        # ── Pad TARGET_SR audio ───────────────────────────────────────────
        pad_44k = (4096 - wav.shape[1] % 4096) % 4096
        wav_44k = F.pad(wav, (0, pad_44k))

        # ── Downsample to MODEL_SR — critical fix ─────────────────────────
        # Model was trained at 16000 Hz. Feeding 44100 gives garbage output.
        wav_16k  = get_resampler(TARGET_SR, MODEL_SR)(wav_44k)   # (2, T_16k)
        pad_16k  = (4096 - wav_16k.shape[1] % 4096) % 4096
        wav_16k  = F.pad(wav_16k, (0, pad_16k))

        # ── ML model receives 16kHz audio ─────────────────────────────────
        with torch.no_grad():
            mix = wav_16k.unsqueeze(0).to(DEVICE)   # (1, 2, T_16k)
            est = model(mix)                        # (1, 4, 2, T_16k)
            est = est.squeeze(0).cpu()              # (4, 2, T_16k)

        # ── Upsample ML stems back to TARGET_SR ──────────────────────────
        S, C, T_16k = est.shape
        est_flat = est.view(S * C, T_16k)
        est_44k  = get_resampler(MODEL_SR, TARGET_SR)(est_flat).view(S, C, -1)
        # est_44k : (4, 2, T_44k)

        # ── DSP impacts — on TARGET_SR audio, fully independent ───────────
        dsp_impacts, impact_mask, n_events = extract_impacts_dsp(
            wav_44k,
            sr           = TARGET_SR,
            ratio_thresh = 8.0,
            energy_floor = 1e-5,
            min_dur_ms   = 15.0,
            post_roll_ms = 150.0,
        )

        # Align lengths (upsample rounding ±few samples)
        T_out = est_44k.shape[2]
        if dsp_impacts.shape[1] > T_out:
            dsp_impacts  = dsp_impacts[:, :T_out]
            impact_mask  = impact_mask[:T_out]
        elif dsp_impacts.shape[1] < T_out:
            pad = T_out - dsp_impacts.shape[1]
            dsp_impacts = F.pad(dsp_impacts, (0, pad))
            impact_mask = F.pad(impact_mask, (0, pad))

        # ── Final stems ───────────────────────────────────────────────────
        # Speech / Music / Noise : ML model output
        # Impacts                : DSP output  (ML impacts index 2 discarded)
        final_stems = {
            'speech':  est_44k[0],      # ML
            'music':   est_44k[1],      # ML
            'impacts': dsp_impacts,     # DSP
            'noise':   est_44k[3],      # ML
        }

        # ── Save stems ────────────────────────────────────────────────────
        stem_urls = {}
        for key, stem_wav in final_stems.items():
            peak = stem_wav.abs().max()
            if peak > 1.0:
                stem_wav = stem_wav / peak * 0.99
            fname    = f"{job_id}_{key}.wav"
            out_path = os.path.join(STEM_DIR, fname)
            torchaudio.save(out_path, stem_wav.contiguous(), TARGET_SR)
            stem_urls[key] = f"/stems/{fname}"

        orig_fname = f"{job_id}_original.wav"
        torchaudio.save(os.path.join(STEM_DIR, orig_fname), wav_44k, TARGET_SR)

        impact_pct = float(impact_mask.mean().item() * 100)

        return jsonify({
            "stems":    stem_urls,
            "original": f"/stems/{orig_fname}",
            "job_id":   job_id,
            "dsp_info": {
                "impact_events_detected":   n_events,
                "impact_coverage_pct":      round(impact_pct, 2),
                "impacts_method":           "DSP transient detector",
                "speech_plosive_protected": True,
            },
            "separation_note": (
                f"Speech/Music/Noise via ML model (16kHz). "
                f"Impacts via DSP — {n_events} event(s), "
                f"{impact_pct:.1f}% of audio flagged."
            )
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
