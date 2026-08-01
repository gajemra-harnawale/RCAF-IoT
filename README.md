<div align="center">

# RCAF-IoT

### A Lightweight RGB-Depth Context-Aware Anomaly Fusion Engine
### for Resource-Constrained IoT Edge Nodes

[![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=flat-square)]()
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-c51a4a?style=flat-square)]()
[![Sensor](https://img.shields.io/badge/Sensor-Microsoft%20Kinect%20v1-00599C?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-VJTI%20Mumbai-darkgreen?style=flat-square)]()

<br/>

*Department of Electronics Engineering*
*Veermata Jijabai Technological Institute (VJTI)*
*Mumbai 400019, Maharashtra, India*

</div>

---

## Overview

RCAF-IoT is a multimodal anomaly detection research
framework designed and validated on Raspberry Pi 4
ARM Cortex-A72 edge nodes equipped with Microsoft
Kinect Xbox 360 Model 1473 (Kinect v1) sensors.

The system fuses depth and network features into a
48-dimensional context-aware representation processed
by a hybrid ensemble of supervised and unsupervised
anomaly detectors. A Depth-Silhouette Privacy Module
provides differential privacy guarantees for biometric
depth data. A STRIDE-DREAD threat model enumerates
security threats specific to this trimodal IoT
architecture.

---

## Hardware Platform

| Component | Specification |
|:----------|:-------------|
| Edge Node | Raspberry Pi 4B |
| Processor | ARM Cortex-A72 @ 1.8GHz (4 cores) |
| RAM | 4 GB LPDDR4 |
| Sensor | Microsoft Kinect Xbox 360 Model 1473 |
| Depth Resolution | 320×240 @ 30fps (800–4000mm) |
| RGB Resolution | 640×480 @ 30fps |
| Audio | 16kHz 4-microphone array |
| OS | Raspberry Pi OS 64-bit (Debian Bookworm) |

---

## Key Results

### Detection Performance (5-Fold Cross-Validation)

| Metric | Mean | Std | Min | Max |
|:-------|-----:|----:|----:|----:|
| F1 Score | 0.9991 | 0.0001 | 0.9989 | 0.9992 |
| Detection Rate | 0.9997 | 0.0002 | 0.9995 | 1.0000 |
| Precision | 0.9985 | 0.0001 | 0.9983 | 0.9986 |
| ROC-AUC | 0.9997 | 0.0001 | 0.9996 | 0.9998 |
| False Positive Rate | 0.0017 | 0.0001 | 0.0016 | 0.0019 |

### ARM Hardware Benchmarks

| Metric | Value |
|:-------|------:|
| Avg Inference Latency | 45.2 ms |
| Real-time Target (<100ms) | ✓ MET |
| Throughput | 22.1 samples/sec |

### Baseline Comparison

| Method | F1 | DR | FPR | AUC |
|:-------|---:|---:|----:|----:|
| Isolation Forest | 0.539 | 0.517 | 0.459 | 0.529 |
| One-Class SVM | 0.400 | 0.268 | 0.085 | 0.592 |
| Local Outlier Factor | 0.747 | 0.877 | 0.538 | 0.669 |
| Random Forest | 0.999 | 1.000 | 0.001 | 0.999 |
| Gradient Boosting | 0.999 | 0.999 | 0.001 | 0.999 |
| **RCAF-IoT v2 (Ours)** | **0.999** | **0.999** | **0.001** | **0.999** |

### STRIDE-DREAD Threat Model

| Layer | Threats | Critical | High | Medium |
|:------|--------:|---------:|-----:|-------:|
| Device | 6 | 1 | 4 | 1 |
| Network | 6 | 5 | 1 | 0 |
| Application | 7 | 0 | 7 | 0 |
| **Total** | **19** | **6** | **12** | **1** |
| **Avg DREAD Score** | **7.21** | | | |

### Privacy Module (Depth-Silhouette Anonymisation)

| ε | PSNR (dB) | Occupancy | Latency (ms) |
|:-:|----------:|----------:|-------------:|
| 0.1 | 24.25 | 0.449 | 19.8 |
| 0.5 | 32.03 | 0.992 | 18.9 |
| **1.0** | **33.35** | **1.000** | **18.9** |
| 2.0 | 33.82 | 1.000 | 19.0 |
| 5.0 | 33.90 | 1.000 | 19.0 |
| 10.0 | 33.93 | 1.000 | 19.0 |

---

## Research Contributions

| | Contribution |
|:-|:-------------|
| **C1** | First STRIDE-DREAD threat model for trimodal Kinect v1 + Raspberry Pi 4 IoT edge architecture (19 threat vectors, 3 layers, avg DREAD 7.21) |
| **C2** | RCAF-IoT: 48-dimensional depth-weighted context-aware feature fusion with RF + GB + IF hybrid ensemble achieving F1 = 0.9991 at 45.2ms on ARM |
| **C3** | Depth-Silhouette Privacy Module: first quantified ε-PSNR tradeoff for Kinect v1 biometric depth anonymisation using Gaussian DP mechanism |
| **C4** | First ARM-validated inference benchmarks for multimodal IoT IDS on Raspberry Pi 4 under live sensor load |

---

## System Architecture

```text
Microsoft Kinect v1 (Model 1473)
RGB-D Sensor
│
┌──────────────────┴──────────────────┐
│                                     │
▼                                     ▼
RGB Stream                       Depth Stream
(640×480 @ 30 fps)               (320×240 @ 30 fps)
│                                     │
▼                                     ▼
RGB Preprocessing                Depth Preprocessing
- Resize                         - Filtering
- Denoising                      - Normalization
- Normalization                  - Hole Filling
│                                     │
▼                                     ▼
RGB Feature Extraction           Depth Feature Extraction
(16-D Features)                  (32-D Features)
│                                     │
└─────────────────┬───────────────────┘
                  │
                  ▼
     Weighted Feature Fusion Module
     (RGB: 16-D + Depth: 32-D x Higher Weight)
                  │
                  ▼
       48-Dimensional Feature Vector
                  │
     ┌────────────┼────────────┐
     │            │            │
     ▼            ▼            ▼
Random Forest  Gradient    Isolation
Classifier     Boosting     Forest
               Classifier   Detector
Weight=0.45   Weight=0.45  Weight=0.10
     │            │            │
     └────────────┴────────────┘
                  │
                  ▼
       Weighted Ensemble Voting
                  │
                  ▼
         Final Decision Module
    ┌──────────────┴──────────────┐
    ▼                             ▼
Normal Activity           Anomalous Activity
```

---

## Repository Structure

```text
RCAF-IoT/
│
├── scripts/
│   ├── dataset_preprocessor.py
│   ├── rcaf_engine_v2.py
│   ├── baseline_comparison.py
│   ├── statistical_validation.py
│   ├── privacy_module.py
│   ├── stride_dread_model.py
│   └── generate_figures.py
│
├── results/
│   ├── rcaf_v2_results.json
│   ├── baseline_comparison.json
│   ├── cross_validation_results.json
│   ├── privacy_module_evaluation.json
│   ├── stride_dread_threat_model.json
│   └── stride_dread_threat_model.csv
│
├── figures/
│   ├── fig1_privacy_tradeoff.pdf
│   ├── fig2_baseline_comparison.pdf
│   ├── fig3_cross_validation.pdf
│   ├── fig4_stride_dread.pdf
│   └── fig5_latency.pdf
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dependencies

```text
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
matplotlib>=3.6.0
psutil>=5.9.0
cryptography>=40.0.0
```

---

## Datasets

| Dataset | Source | Description |
|:--------|:-------|:------------|
| UCSD Anomaly Detection v1p2 | [SVCL UCSD](http://www.svcl.ucsd.edu/projects/anomaly/) | Real pedestrian footage (18,560 frames) |
| KDD Cup 1999 | [UCI ML Repository](http://kdd.ics.uci.edu/databases/kddcup99/) | Network traffic features (41-dim) |

---

## Figures

| Figure | Description |
|:-------|:------------|
| fig1_privacy_tradeoff | ε-PSNR privacy-utility tradeoff curve |
| fig2_baseline_comparison | Detection performance vs 5 baselines |
| fig3_cross_validation | 5-fold CV box plots |
| fig4_stride_dread | STRIDE-DREAD risk matrix (19 threats) |
| fig5_latency | ARM inference latency benchmark |

---

## Related Repositories

- [SENTINEL-IoT](https://github.com/gajemra-harnawale/SENTINEL-IoT)
- [GUARDIAN-IoT](https://github.com/gajemra-harnawale/GUARDIAN-IoT)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

| Field | Details |
|:------|:--------|
| Email | gcharnawale_p25@el.vjti.ac.in |
| Department | Electronics Engineering |
| Institution | Veermata Jijabai Technological Institute (VJTI) |
| Location | Mumbai 400019, Maharashtra, India |

---

<div align="center">

*This work is part of ongoing PhD research at VJTI Mumbai.*
*Code and results will be fully released upon paper acceptance.*

</div>
