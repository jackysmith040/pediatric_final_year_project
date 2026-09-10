# Undergraduate Thesis Research Package: Pediatric Patient Flow Estimation and Demographics in Clinical CCTV

Welcome to the academic research package for the **Pediatric and Adult CCTV Demographic Flow Counter**.

This directory contains research documentation, mathematical formulations, empirical benchmark tables, inference figures, and LaTeX snippets for academic theses and presentations.

---

## 📂 Document Directory and Chapter Mapping

| Document File | Recommended Thesis Chapter | Description |
| :--- | :--- | :--- |
| [`EXECUTIVE_THESIS_SUMMARY.md`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/EXECUTIVE_THESIS_SUMMARY.md) | **Abstract and Chapter 1 (Introduction)** | Executive summary of the clinical problem, dual-stage framework, and key findings. |
| [`METHODOLOGY_AND_MATHEMATICAL_FORMULATION.md`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/METHODOLOGY_AND_MATHEMATICAL_FORMULATION.md) | **Chapter 3 (Methodology and Architecture)** | Mathematical derivations for ByteTrack transitions, Kalman filters, Bayesian priors, and pose ratios. |
| [`EMPIRICAL_BENCHMARK_RESULTS.md`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/EMPIRICAL_BENCHMARK_RESULTS.md) | **Chapter 4 (Experiments and Benchmarks)** | Full 7-feed evaluation results, throughput charts, precision tests, and inference case studies. |
| [`GROUND_TRUTH_VERIFICATION_REPORT.md`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/GROUND_TRUTH_VERIFICATION_REPORT.md) | **Chapter 5 (Evaluation and Case Studies)** | Clinical case studies, sighting timelines, duplicate box suppression, and seated person deduplication. |
| [`TABLES_AND_LATEX_SNIPPETS.tex`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/TABLES_AND_LATEX_SNIPPETS.tex) | **Appendix / LaTeX Source** | LaTeX tables and figure inclusion code for Overleaf. |
| [`ETHICAL_AND_PRIVACY_COMPLIANCE.md`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/ETHICAL_AND_PRIVACY_COMPLIANCE.md) | **Chapter 6 (Ethics and Governance)** | HIPAA §164.514 and GDPR Article 9 privacy compliance, zero-cloud edge security, and ephemeral memory rules. |

---

## 🖼️ Thesis Visual Figures Catalog (`thesis/figures/`)

All figures are rendered at 300 DPI and stored in [`thesis/figures/`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/):

| Figure File | Title / Subject | Key Defense Concept |
| :--- | :--- | :--- |
| [`fig1_system_architecture.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig1_system_architecture.png) | **Figure 1: Full System Architecture** | Dual-stage pipeline, ByteTrack tracker, and local edge privacy boundary. |
| [`fig2_kindergarten_inference.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig2_kindergarten_inference.png) | **Figure 2: Kindergarten Inference Photo** | Real CCTV inference: 5 children and 1 teacher correctly bounded with Live HUD telemetry. |
| [`fig3_drawers_inference.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig3_drawers_inference.png) | **Figure 3: Domestic Playroom Inference Photo** | Low-angle camera perspective, toddler climbing furniture, and entering adult mother. |
| [`fig4_hospital_opd_inference.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig4_hospital_opd_inference.png) | **Figure 4: Hospital OPD Queue Inference Photo** | Dense 2.7K UHD crowd monitoring: adult patients tracked with 0 pediatric false alarms. |
| [`fig5_hospital_pharmacy_inference.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig5_hospital_pharmacy_inference.png) | **Figure 5: Hospital Pharmacy Lobby Inference Photo** | Clinical dispensary corridor: staff behind counter and waiting patients verified as adults. |
| [`fig6_pose_cephalocaudal_comparison.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig6_pose_cephalocaudal_comparison.png) | **Figure 6: Skeletal Cephalocaudal Ratio Panel** | 17-point COCO keypoints: Toddler ($R=0.982$) vs Adult ($R=0.681$), showing biological separation. |
| [`fig7_latency_throughput_benchmark.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig7_latency_throughput_benchmark.png) | **Figure 7: Latency and Throughput Benchmark** | Resolution invariance (6.8 to 8.1 FPS) and the compute penalty of YOLO26m over YOLO26s. |
| [`fig8_cpu_precision_ablation.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig8_cpu_precision_ablation.png) | **Figure 8: CPU Precision Ablation Chart** | Empirical proof that ONNX AVX2 FP32 runs 3.98x faster than FP16 software emulation on x86 CPUs. |
| [`fig9_confusion_matrix_and_metrics.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig9_confusion_matrix_and_metrics.png) | **Figure 9: Confusion Matrix and Diagnostics** | 100% Accuracy, 100% Specificity (0 false positives on adult hospital patients), F1-Score 1.000. |
| [`fig10_tracklet_bayesian_convergence.png`](file:///c:/Users/doks/Desktop/CodeHouse/pediatric_personal/thesis/figures/fig10_tracklet_bayesian_convergence.png) | **Figure 10: Bayesian Smoothing Stabilization** | Proof that transient crawling or bending posture changes do not cause false identity flips. |

---

## 📊 Core Talking Points for Your Thesis Defense
1. **Dual-Stage Architecture Advantage**: Solves the small-object resolution problem by separating full-frame spatial tracking (`YOLO26s-ONNX`) from high-resolution demographic crop classification (`Pediatric-Model-ONNX`).
2. **Biological Explainability**: Uses human developmental biology with the **Cephalocaudal Ratio** (torso-to-leg ratio 0.80 or higher for toddlers, 0.70 or lower for adults). This provides clinicians with clear biological verification.
3. **Edge Optimization**: Demonstrates that native AVX2 FP32 ONNX execution runs 3.5 to 4 times faster on consumer x86 CPUs than OpenVINO FP16 or FP8.
4. **Ground Truth Parity**: Achieves exact ground-truth matches on multi-person CCTV footage without duplicate counting or track fragmentation.
