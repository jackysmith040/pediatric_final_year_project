# Ethical AI, Patient Privacy, and Healthcare Compliance (HIPAA / GDPR)

## 1. Introduction and Ethical Statement
Operating computer vision systems in hospital outpatient departments, waiting rooms, and clinical corridors introduces privacy and ethical requirements. Handling video of vulnerable pediatric patients requires compliance with data protection laws:
- **HIPAA (Health Insurance Portability and Accountability Act)**: Protection of Protected Health Information (PHI).
- **GDPR (General Data Protection Regulation)**: Articles 9 and 35 governing biometric processing and Data Protection Impact Assessments (DPIA).

This document describes the privacy-by-design principles implemented in our dual-stage architecture.

---

## 2. Privacy-by-Design Architecture

```
  ┌────────────────────────────────────────────────────────┐
  │ CCTV Camera Sensor (Local RTSP / USB DirectShow)       │
  └──────────────────────────┬─────────────────────────────┘
                             │ Local Uncompressed Frame
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Edge Inference Processing Unit (Local On-Premises CPU) │
  │ - In-Memory Ephemeral Frame Consumption                │
  │ - Transient 224x224 Anonymized Bounding Box Crops      │
  │ - Non-Biometric Skeletal Coordinates (x, y keypoints)  │
  └──────────────────────────┬─────────────────────────────┘
                             │ Numerical Aggregated Counts Only
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Anonymized Metadata & Triage Analytics Store           │
  │ - Count: { "children": 5, "adults": 1, "timestamp" }   │
  │ - Zero Facial Embeddings, Zero Identity Vectors        │
  │ - HIPAA / GDPR Compliant by Construction               │
  └────────────────────────────────────────────────────────┘
```

---

## 3. Core Compliance Mechanisms

### 3.1 Zero-Cloud and Local Edge Execution
* **Threat Model**: Eavesdropping, interception, or third-party cloud data leaks.
* **Mitigation**: All detection, tracking, classification, and pose estimation run locally on hospital hardware. The system never transmits video streams, frame buffers, or image crops over the public internet.

### 3.2 Ephemeral Frame Processing
* Raw video frames and cropped regions exist only in volatile system RAM during the inference window (~120 ms).
* The pipeline discards frames immediately after updating track states. The system does not create or store permanent facial databases or identifiable biometric records.

### 3.3 De-Identification via Abstract Skeletal Keypoints
* The pose estimation plugin represents human bodies as 17 spatial coordinates:
  $$\mathbf{k}_j = (x_j, y_j) \in \mathbb{R}^2$$
* These coordinates measure anatomical proportions (torso-to-leg ratio). They do not capture facial geometry, skin texture, race, or identifiable physical features. Under GDPR Recital 26, anonymous statistical information that does not identify a natural person is exempt from data protection restrictions.

### 3.4 Explainable Audit Trails
* Instead of opaque decisions, all demographic classifications produce verifiable audit trails:
  - Bounding box scale ratio ($h_{\text{norm}}$)
  - Bayesian historical confidence consensus ($\bar{P}$)
  - Cephalocaudal skeletal ratio ($R_{\text{ceph}}$)
* Clinical staff and hospital administrators can inspect the reasoning timeline without storing raw patient video recordings.
