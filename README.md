# 🧠 Deepfake Detection Using Physiological rPPG Signals (No Training Required)

## Table of Content

| Section | Description |
|--------|-------------|
| Introduction | Overview of deepfake threats and motivation for physiological detection |
| Problem Statement | Why existing deepfake detectors fail against unseen attack types |
| Why Physiological rPPG-Based Detection? | Justification for using biological signals instead of trained datasets |
| System Overview | High-level architecture of the rPPG deepfake detection system |
| Workflow | Step-by-step pipeline from video input to final verdict |
| Key Features | Summary of major technical contributions and enhancements |
| Signal Processing Pipeline | Detailed explanation of rPPG extraction and processing stages |
| Physiological Parameters Explained | Explanation of heartbeat-related metrics used for detection |
| Thresholds and Decision Logic | Exact thresholds and scoring rules for REAL vs FAKE verdict |
| Suspicious Segment Detection | How temporal inconsistencies and artifacts are identified |
| Visualization Outputs | Interpretation of waveform and frequency spectrum plots |
| Example Output Explanation | Line-by-line explanation of a sample API response |
| Limitations | Known constraints and failure cases |
| Robustness Against New Attacks | Why biological signals generalize across deepfake methods |
| Future Enhancements | Planned improvements and research directions |
| How to Run the Project | Instructions to execute the FastAPI-based system |
| Conclusion | Summary of findings and impact |
| References | Research papers and technical sources |

---

## 🔍 Project Overview

This project detects **deepfake videos** by analyzing **physiological signals**
(remote Photoplethysmography – rPPG) extracted from facial skin regions.

Instead of relying on **trained datasets**, **CNNs**, or **attack-specific artifacts**,
this system uses **biological consistency rules** such as heart rhythm, signal strength,
spectral purity, and temporal regularity to classify videos as **REAL** or **FAKE**.

✅ No datasets  
✅ No training or fine-tuning  
✅ Attack-agnostic  
✅ Explainable decision logic  

---

## ❓ Why Not Traditional Deepfake Detectors?

Most deepfake detection models today:
- Are trained on **specific datasets**
- Learn **visual artifacts** (blur, texture mismatch, compression noise)
- Perform poorly on **unseen or future attack methods**

### Example:
- A model trained on **FaceSwap (2018)** often fails on
  **Diffusion-based or GAN-v4 deepfakes (2024–2025)**.

### Core Problem:
> **These models learn attacks, not human biology**

---

## 🫀 Why rPPG-Based Detection Works

Every real human face naturally exhibits:
- Blood flow under the skin
- Rhythmic color changes
- Physiological constraints (heart rate range, regularity)

Deepfake generation pipelines:
- Do **not simulate real blood perfusion**
- Cannot maintain **temporal physiological consistency**
- Produce texture flicker instead of true pulse signals

🧬 **Biology is invariant to attack type**  
Even if generation methods change,  
**human physiology does not.**

---

## ⚙️ System Workflow

Input Video

↓

Face Detection (MediaPipe FaceMesh)

↓

Skin ROI Extraction (Forehead + Cheeks)

↓

RGB Signal Collection (per frame)

↓

rPPG Extraction (POS Method)

↓

Signal Processing (Detrend + Bandpass Filter)

↓

Feature Extraction (Time + Frequency Domain)

↓

Physiological Rule Validation

↓

Confidence Scoring

↓

FINAL VERDICT → REAL / FAKE


---

## 🧪 rPPG Extraction Method Used

### POS (Plane-Orthogonal-to-Skin)

- Uses normalized RGB color changes
- Projects signals onto a plane orthogonal to skin tone
- Cancels lighting variations
- Enhances pulse-related chromatic changes

**Why POS?**
- Robust to illumination changes
- No training required
- Widely validated in medical rPPG literature

---

## 📊 Extracted Physiological Parameters

### 1. Signal Strength (Standard Deviation)

**What it means:**  
Measures how much the rPPG signal fluctuates.

**Why it matters:**  
Real blood flow causes controlled color variation.

**Thresholds:**
- < 0.01 → No signal / fake / bad capture
- Ideal (Real): 0.15 – 0.40

---

### 2. Signal-to-Noise Ratio (SNR)

**What it means:**  
Strength of the dominant heartbeat frequency vs background noise.

**Why it matters:**  
This is the **most important deepfake indicator**.

**Thresholds:**
- Noise: < 0.5
- Acceptable: > 1.0
- Ideal (Real): > 2.0

---

### 3. Heart Rate Validity (BPM)

**What it means:**  
Estimated heart rate from dominant frequency.

**Why it matters:**  
Enforces biological plausibility.

**Valid Range:**
- 50 – 135 BPM  
(Only trusted if SNR > 0.8)

---

### 4. Spectral Concentration (HR Power Ratio)

**What it means:**  
Percentage of signal energy inside the heart-rate band (0.7–4.0 Hz).

**Why it matters:**  
Real pulse concentrates energy in a narrow band.

**Thresholds:**
- Noise: < 0.10
- Ideal (Real): > 0.30

---

### 5. Peak Regularity

**What it means:**  
Consistency of time gaps between heartbeats.

**Why it matters:**  
Real hearts beat rhythmically; deepfakes flicker.

**Thresholds:**
- Irregular: < 0.2
- Ideal: 0.5 – 0.8

---

### 6. Spectral Purity

**What it means:**  
How clean the main heartbeat peak is.

**Why it matters:**  
Real signals have one dominant peak + harmonics.

**Thresholds:**
- Messy spectrum: < 0.25
- Ideal: > 0.50

---

### 7. Motion Penalty

**What it means:**  
Measures facial motion using optical flow.

**Why it matters:**  
Heavy motion corrupts rPPG signals.
A “perfect” heartbeat during heavy motion is suspicious.

**Penalty Applied If:**  
Motion score > 0.8

---

### 8. Suspicious Segments

**What it means:**  
Specific time intervals that look artificial.

**Detected Using:**
- Flat signal regions
- Sudden amplitude jumps
- Abnormal variance spikes

**Why it matters:**  
GAN-based deepfakes often fail temporally.

---

## 🧠 Final Verdict Logic

Each metric contributes points (Max = 100):

- Window consistency
- Signal quality
- Physiological validity
- Cross-region coherence
- Motion penalty

### Decision Rule:
- **Score > 45% → REAL**
- **Score ≤ 45% → FAKE**

---

## 🧪 Example Output (Simplified)


Heart Rate: 53.2 BPM

SNR: 5.45

Signal Strength: 0.026

Peak Regularity: 0.78

Spectral Concentration: 0.47

Final Verdict: REAL

Confidence: 66%


---

## 🚀 Why This Approach Is Future-Proof

| Dataset Models | This System |
|---------------|------------|
| Learns attacks | Learns biology |
| Breaks on new methods | Attack-agnostic |
| Black-box | Fully explainable |
| Requires retraining | Zero training |

---

## 📌 Intended Use

- Academic projects
- Research papers
- Explainable AI demonstrations
- Deepfake robustness analysis
- Real-world verification pipelines

---

## 🏁 Conclusion

This project proves that **deepfake detection does not require training data**
when **physiological truth** is used as the ground reference.

> Attacks evolve.  
> Biology does not.

---

