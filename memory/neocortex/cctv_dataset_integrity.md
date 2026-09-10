---
neuron_id: cctv-dataset-integrity
title: CCTV Dataset Container Integrity and Shift Indexing
synaptic_weight: 45
corpus_callosum: orchestrator_ai
blindspot: false
summary: Explains the container divergence between valid Dahua IMKH media streams and empty zero-filled download placeholders in hospital CCTV datasets.
---

# CCTV Dataset Container Integrity and Shift Indexing

## Context
In hospital surveillance analysis, raw MP4 exports may contain either complete proprietary containers or pre-allocated zero-byte download placeholders.

## Discovery
1. Valid Dahua/Hikvision CCTV exports in this dataset feature the `IMKH` container header.
2. Interrupted or unfinalized exports appear as large 1024 MB files filled entirely with null bytes (`0x00`), which fail with `moov atom not found`.
3. Valid feeds correspond to discrete hospital shifts (Morning, Midday, Afternoon, Evening) spanning 732,816 real frames.
