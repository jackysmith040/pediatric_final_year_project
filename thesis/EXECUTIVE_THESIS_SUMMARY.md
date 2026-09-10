# Executive Thesis Summary: Explainable Edge-AI for Pediatric Patient Flow Estimation in Clinical CCTV Environments

**Author / Researcher**: Pediatric Personal Research Project  
**Domain**: Computer Vision, Healthcare Informatics, Edge AI, and Biometric Demographics  
**Target Application**: Automated Pediatric vs. Adult Patient Flow Auditing, Clinic Overcrowding Prevention, and Hospital Triage Analytics  

---

## 1. Abstract
Accurate quantification of pediatric patient flow in outpatient departments and clinical spaces helps hospitals allocate staff and manage infection risks. Standard object detection models often fail in closed-circuit television (CCTV) feeds for three reasons:
1. **Scale Ambiguity**: Toddlers sitting or crawling are often confused with objects or seated adults.
2. **Track Fragmentation**: Crowded queues and posture changes (sitting to standing) cause duplicate counts.
3. **Hardware and Privacy Constraints**: Hospital regulations prohibit sending unencrypted video to cloud servers. The system must run on local commodity x86 CPUs without dedicated GPUs.

This thesis presents a two-stage neuro-symbolic framework for CPU edge hardware. Stage 1 uses a person detector (`YOLO26s-ONNX`, 38.8 MB) paired with a ByteTrack state machine to produce stable tracks at 7.5 to 8.1 frames per second. Stage 2 passes standardized crops to a fine-tuned classifier (`Pediatric-Model-ONNX`), modulated by Bayesian height priors and an optional 17-keypoint torso-to-leg ratio check (0.80 or higher for toddlers, 0.70 or lower for adults).

In tests across real CCTV feeds from 270p to 2.7K Ultra-HD, the system achieved 100 percent counting accuracy on validation feeds: 5 children and 1 adult in classrooms, 2 children and 1 adult in playrooms, 6 adults in pharmacy halls, and 2 adults in outpatient clinics. The software runs without cloud access and maintains zero display lag.

---

## 2. Key Contributions

### 1. Two-Stage Decoupled Pipeline
The architecture separates spatial localization (Stage 1) from demographic classification (Stage 2). The base detector scans full frames for candidate persons. The demographic model evaluates standardized 224 by 224 pixel crops using Bayesian consensus smoothing over time.

### 2. Anthropometric and Cephalocaudal Validation
The method uses two biological measurements:
- **Bayesian Height Prior**: Environment thresholds ($h_{\text{norm}} \ge 0.32$) distinguish standing adults from stretching or jumping children.
- **Cephalocaudal Ratio ($R_{\text{ceph}}$)**: Young toddlers have large torsos and short legs ($R_{\text{ceph}} = 0.982$). Adults have elongated legs ($R_{\text{ceph}} = 0.62 - 0.74$). This ratio provides a biological test independent of clothing colors.

### 3. Spatial and Temporal Track Stitching
To prevent track splitting when people sit down or move behind furniture, the state machine provides:
- **History-Prioritized Containment**: Prevents newly spawned bounding boxes from overwriting established tracks.
- **Lost-Tracklet Stitching**: Matches candidate tracks across time gaps up to 30 seconds using spatial overlap and height consistency.
- **Anatomical Body-Part Deduplication**: Merges adjacent head, torso, and knee bounding boxes for seated individuals into a single track.

### 4. CPU Hardware Optimization
The study tests inference performance on x86 consumer CPUs:
- **ONNX FP32** runs natively on AVX2 vector registers, completing full video runs 22 seconds faster than PyTorch.
- **Intel OpenVINO FP16** slowed execution down by 3.5 to 4.5 times (438 ms vs 110 ms) because consumer x86 CPUs lack native 16-bit floating-point registers.
- **FP8** cannot execute on consumer CPUs without specialized tensor hardware.

---

## 3. Core Empirical Results Summary

| Benchmark Feed | Native Resolution | Ground Truth | System Count | Counting Accuracy | Latency (ONNX CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Child Room Drawers** | 480 × 270 | 2 Kids, 1 Adult | **2 Kids, 1 Adult** | **100%** | **123.3 ms (8.1 FPS)** |
| **Kindergarten Classroom** | 720 × 1280 | 5 Kids, 1 Adult | **5 Kids, 1 Adult** | **100%** | **131.7 ms (7.6 FPS)** |
| **Hospital OPD Morning Shift** | 2688 × 1520 | 0 Kids, 2 Adults | **0 Kids, 2 Adults** | **100%** | **146.6 ms (6.8 FPS)** |
| **Hospital Pharmacy Lobby** | 2688 × 1520 | 0 Kids, 6 Adults | **0 Kids, 6 Adults** | **100%** | **134.0 ms (7.5 FPS)** |

---

## 4. Academic Citation and Chapter Mapping
This document summarizes:
- **Chapter 1**: Introduction and Clinical Motivation (Triage and Overcrowding).
- **Chapter 3**: Two-Stage Architecture and State Machine Formulation.
- **Chapter 4**: Anthropometric and Cephalocaudal Ratios.
- **Chapter 5**: Hardware Benchmarking and Edge Optimization.
- **Chapter 6**: Results, Ground Truth Verification, and HIPAA Compliance.
