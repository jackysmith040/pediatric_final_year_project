---
neuron_id: pediatric-demographic-classification-linkage
title: Pediatric Demographic Classification Linkage and Dynamic Refinement
synaptic_weight: 50
corpus_callosum: evie
blindspot: true
summary: Links YOLO26s base person detector and fine-tuned pediatric demographic model with crop-area coverage filters and dynamic lifecycle state machine counting.
---

# Pediatric Demographic Classification Linkage and Two-Stage Pipeline Integrity

## Architectural Core: Two-Stage Pipeline (2 Dedicated Models)
The system strictly operates as a **Two-Stage Architecture using 2 distinct models**:
1. **Stage 1 (Base Full-Field CCTV Detector & Tracker)**:
   - Model: `computer_vision_models/yolo26/base_model/yolo26s.pt` (**YOLO26s**, COCO class 0: `person`).
   - Tracker: ByteTrack (tracks persistent person IDs across wide CCTV perspectives).
2. **Stage 2 (Demographic Classification on Crops)**:
   - Model: `computer_vision_models/yolo26/fine_tune_model/pediatric-model.pt` (YOLO26 fine-tuned pediatric model).
   - Mechanism: Receives isolated, letterboxed person crops (`_extract_crop`), runs inference, and computes $P(\text{child})$ vs $P(\text{adult})$.

> [!WARNING] Blindspot
> 1. **Model Identity**: The architecture strictly standardizes on **YOLO26s**, not legacy YOLO architectures.
> 2. **Crop Area Coverage**: `pediatric-model.pt` fires `child` on partial adult torsos/limbs unless filtered by crop area coverage ($\text{Area}_{\text{det}} / \text{Area}_{\text{crop}} \ge 0.40$) and scene spatial scale priors.

## Refinement Protocols
1. **Dynamic Metadata Resolution**: `_classify_crop` inspects active model class dictionaries (`model.names`) directly.
2. **Crop Area Coverage Filter**: Discards spurious partial-torso child hits on adult crops.
3. **Dynamic Membership Refinement**: If early track frames are ambiguous, subsequent high-confidence observations dynamically update the track's demographic set without count inflation.
4. **Calibrated Decision Boundary**: Uses environment-aware demographic priors ($H_0 = \text{Adult}$ in Hospital OPD).

