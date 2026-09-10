---
neuron_id: skeletal-pose-anthropometry
title: Skeletal Pose Keypoints & OpenVINO Quantization for Surveillance
synaptic_weight: 46
corpus_callosum: spatial-temporal-tracklet-stitching
blindspot: false
summary: Explains the application of 17-point skeletal pose estimation (torso-to-leg cephalocaudal ratios) to overcome CCTV perspective compression and how OpenVINO FP16 quantization accelerates CPU inference.
---

# Skeletal Pose Keypoints & OpenVINO Quantization for Surveillance

## Conceptual Foundation

1. **Cephalocaudal Developmental Proportions**:
   - Standard 2D bounding boxes conflate posture with physical stature: an adult sitting or slouching exhibits a compressed vertical box indistinguishable from a standing child.
   - 17-point skeletal pose estimation extracts anatomical landmarks independent of camera pitch:
     * **Torso Vector**: $\vec{v}_{\text{torso}} = \frac{\text{Shoulder}_L + \text{Shoulder}_R}{2} - \frac{\text{Hip}_L + \text{Hip}_R}{2}$.
     * **Leg Vector**: $\vec{v}_{\text{leg}} = \text{Hip} - \text{Ankle}$.
   - Human growth follows a cephalocaudal trajectory: toddlers have disproportionately long torsos relative to their lower limbs ($\text{Torso}/\text{Leg} \ge 0.80$), whereas adults have elongated lower limbs ($\text{Torso}/\text{Leg} \le 0.70$).

2. **OpenVINO FP16 Quantization Dynamics**:
   - `dynamic=true, quantize=16` cuts model footprint in half (e.g. 31.5 MB to 16.9 MB) while maintaining dynamic shape flexibility across different CCTV frame dimensions.
   - On x86 CPU vector units (AVX2/F16C and AVX-512 VNNI), 16-bit floating point operations double SIMD register throughput and minimize L3 cache evictions during frame decoding.
