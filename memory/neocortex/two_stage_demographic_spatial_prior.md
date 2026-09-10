---
neuron_id: two-stage-demographic-spatial-prior
title: Two-Stage Demographic Spatial Prior & Slicing Enhancement
synaptic_weight: 45
corpus_callosum: pediatric-demographic-classification-linkage
blindspot: false
summary: Explains the fusion of fine-tuned object detection crops with relative spatial height priors and optional SAHI/CLAHE plugins to resolve demographic ambiguity in CCTV surveillance.
---

# Two-Stage Demographic Spatial Prior & Slicing Enhancement

## Conceptual Foundation
When deploying fine-tuned pediatric classification models on high-angle CCTV feeds, raw crop detection confidences must be differentiated from softmax distributions. In a two-stage architecture:
1. **Stage 1 (Surveillance Person Detector)** localizes all persons regardless of scale.
2. **Stage 2 (Pediatric Crop Evaluator)** identifies child vs adult features inside normalized person crops.
3. **Spatial Height Prior**: In fixed-perspective camera feeds, physical height ratios ($h_{box} / h_{frame}$) provide a strong Bayesian prior. Tall individuals ($h_{ratio} \ge 0.22$ vertical or $\ge 0.38$ horizontal) receive an adult bias, accurately separating adults from toddlers even under challenging camera angles.
4. **Slicing & Contrast Plugins**: Slicing Aided Hyper Inference (SAHI) solves the small-scale token degradation of distant toddlers, while CLAHE solves localized backlighting and shadow obscuration.
