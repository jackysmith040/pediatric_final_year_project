---
neuron_id: hevc-poc-seek-dynamics
title: HEVC Picture Order Count (POC) Non-Keyframe Seek Dynamics
synaptic_weight: 48
corpus_callosum: orchestrator_ai
blindspot: false
summary: Explains the mechanics of H.265/HEVC Picture Order Count warnings during mid-GOP video seeking and deterministic C-level file descriptor suppression.
---

# HEVC Picture Order Count (POC) Non-Keyframe Seek Dynamics

## Context
When analyzing long hospital CCTV streams, users frequently jump (`start_frame: 1450`) to active human movement sections. During seeking, FFmpeg low-level C decoders may emit `[hevc @ ...] Could not find ref with POC 131`.

## Engineering Explanation
1. **POC (Picture Order Count)**: A counter defined in the ITU-T H.265 / HEVC standard specifying the display order of decoded frames.
2. **GOP Inter-frame Compression**: Surveillance video uses Group of Pictures (GOP) comprising I-frames (keyframes), P-frames (forward prediction), and B-frames (bi-directional prediction).
3. **Mid-GOP Seeking**: When OpenCV jumps directly to a non-keyframe index (such as frame 1450), the B/P-frame references past frames (e.g. POC 131 or 0) that were not decoded in that seek jump.
4. **Resolution**: This is a harmless codec diagnostic, not a logic bug. We implemented a clean C-level file descriptor redirection (`os.dup2`) during the initial seek in `VideoReader`, allowing the decoder to settle cleanly while eliminating terminal stderr noise.
