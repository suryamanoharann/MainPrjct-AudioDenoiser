# Advanced Audio Denoiser: A Hybrid Deep Learning and DSP Architecture for Audio Source Separation

## Overview
**Advanced Audio Denoiser** is a high-fidelity audio enhancement platform engineered to isolate human speech from complex, non-stationary environmental noise. By integrating a fine-tuned deep learning backbone with precision Digital Signal Processing (DSP), the system successfully disentangles Speech, Music, Impacts, and ambient Noise into independent audio tracks. 

This hybrid approach effectively circumvents the temporal smearing of transient sounds often observed in purely neural-network-based architectures, yielding professional-grade acoustic reconstruction and an interactive, browser-based digital mixing environment.

---

## Core Capabilities

* **Precision Denoising & 4-Stem Separation:** Deconstructs complex audio mixtures into four distinct stems, facilitating the surgical extraction of clean speech by isolating and suppressing background interference.
* **Hybrid ML-DSP Pipeline:** * **Machine Learning Pathway:** Utilizes a custom-tuned **Meta Hybrid Demucs** neural network to process continuous acoustic elements (Speech, Music, and Noise).
  * **Digital Signal Processing Pathway:** Employs algorithmic energy envelope thresholding and spectral centroid checking to extract unpredictable, high-frequency physical impacts without relying on the neural network, thereby ensuring zero temporal smearing.
* **Parametric Web Dashboard:** A responsive, Flask-based GUI acting as a digital mixing console, granting users independent parametric control over the separated stems in real-time.
* **Dynamic Heuristic Interface:** Incorporates Auto-Hiding Logic driven by Root Mean Square (RMS) energy thresholding. The system dynamically conceals interface controls for stems that are mathematically absent from the source recording, optimizing the user experience.
* **High-Fidelity Export Options:** Enables the isolated export of individual stems or the rendering of a custom-mixed, denoised master audio file directly to local storage for downstream post-production.

---

## System Architecture & Training Paradigm

* **Neural Backbone:** Fine-tuned `hdemucs_high` architecture utilizing a **Partial Encoder Unfreeze** strategy to leverage generalized pretrained spectral representations while adapting to distinct 4-stem targets.
* **Stochastic Data Pipeline:** Trained on billions of dynamic data variations generated in real-time via a custom `SonicMixer4Stem` loader, integrating the massive **FSD50K** and **LibriSpeech** corpora.
* **Objective Function:** Optimized using a bespoke `hybrid_loss_4stem` criterion, which aggregates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), multi-resolution STFT, and standard L1 loss.
* **Cloud Infrastructure & Continuity:** Engineered an automated **"Baton Pass" checkpointing system** to seamlessly serialize and transfer model states across cloud sessions, successfully bypassing strict GPU hardware limitations during the 61-epoch training cycle.

---

## Evaluation Metrics

Evaluated on a completely unseen test subset, the optimized model achieved significant positive decibel gains, ensuring professional-grade vocal isolation alongside stable background suppression.

| Target Stem | Mean SI-SDR (dB) | Standard Deviation | Performance Assessment |
| :--- | :--- | :--- | :--- |
| **Speech** | +7.16 dB | ±5.49 | Highly Reliable |
| **Music** | +2.21 dB | ±13.53 | Stable Suppression |
| **Noise** | +1.46 dB | ±8.02 | Stable Suppression |
| **GLOBAL MEAN** | **+2.10 dB** | **—** | **Validated System Success** |

*(Note: The Impacts stem is handled exclusively by the parallel DSP pipeline during deployment to guarantee accurate transient extraction and maximum speech clarity).*

---

## Technical Stack

* **Machine Learning / Deep Learning:** Python 3.12, PyTorch 2.0+, Torchaudio
* **Digital Signal Processing:** Librosa, NumPy
* **Backend Application Routing:** Flask
* **Frontend Architecture:** HTML5, CSS3, JavaScript (Web Audio API)

---

## Installation & Deployment

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/Advanced-Audio-Denoiser.git](https://github.com/yourusername/Advanced-Audio-Denoiser.git)
cd Advanced-Audio-Denoiser
