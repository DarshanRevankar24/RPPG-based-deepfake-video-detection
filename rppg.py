import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import os

# -------------------- MediaPipe --------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------------------- Advanced Filtering --------------------
def bandpass_filter(signal, fps, low_hz=0.7, high_hz=4.0, order=4):
    """Enhanced bandpass filter with higher order for better attenuation"""
    nyquist = fps / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    
    low = max(0.01, min(low, 0.99))
    high = max(0.01, min(high, 0.99))
    
    if low >= high:
        low = high - 0.1
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detrend_signal(signal):
    """Remove linear trend from signal"""
    from scipy.signal import detrend
    return detrend(signal)

# -------------------- POS Method (Plane-Orthogonal-to-Skin) --------------------
def apply_pos_method(rgb_signals):
    rgb_signals = np.array(rgb_signals)
    
    if len(rgb_signals) < 10:
        return None
    
    mean_rgb = np.mean(rgb_signals, axis=0)
    if np.any(mean_rgb == 0):
        return None
    
    normalized = rgb_signals / mean_rgb

    S = np.array([[0, 1, -1], [-2, 1, 1]])
    
    try:
        P = np.dot(S, normalized.T)
        
        std_s1 = np.std(P[0, :])
        std_s2 = np.std(P[1, :])
        
        if std_s2 == 0:
            return P[0, :]
        
        alpha = std_s1 / std_s2
        pulse = P[0, :] - alpha * P[1, :]
        
        return pulse
    except:
        return None

# -------------------- CHROM Method --------------------
def apply_chrom_method(rgb_signals, fps):
    rgb_signals = np.array(rgb_signals)
    
    if len(rgb_signals) < 10:
        return None
    
    # Normalize
    mean_rgb = np.mean(rgb_signals, axis=0)
    if np.any(mean_rgb == 0):
        return None
    
    normalized = rgb_signals / mean_rgb
    
    # Calculate chrominance signals
    Xs = 3 * normalized[:, 0] - 2 * normalized[:, 1]
    Ys = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
    
    # Bandpass filter
    Xf = bandpass_filter(Xs, fps)
    Yf = bandpass_filter(Ys, fps)
    
    # Calculate alpha
    std_x = np.std(Xf)
    std_y = np.std(Yf)
    
    if std_y == 0:
        return Xf
    
    alpha = std_x / std_y
    pulse = Xf - alpha * Yf
    
    return pulse

# -------------------- ROI Extraction with Better Handling --------------------
def get_roi_from_landmarks(frame, landmarks, indices):
    h, w, _ = frame.shape
    xs, ys = [], []

    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    if len(xs) < 3 or len(ys) < 3:
        return None

    x1, x2 = max(min(xs), 0), min(max(xs), w)
    y1, y2 = max(min(ys), 0), min(max(ys), h)
    
    # Ensure minimum ROI size
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return frame[y1:y2, x1:x2]

def extract_rgb_from_roi(roi):
    if roi is None or roi.size == 0:
        return None
    
    # Convert to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    mean_rgb = np.mean(roi_rgb.reshape(-1, 3), axis=0)
    if mean_rgb[0] < mean_rgb[2]:
        return None
    
    return mean_rgb

# -------------------- Motion Analysis --------------------
def calculate_motion_robustly(prev_gray, gray):
    """Enhanced motion estimation with error handling"""
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)
    except:
        return 0.0

# -------------------- Advanced Feature Extraction --------------------
def extract_advanced_features(signal, fps):
    features = {}
    
    if len(signal) < fps * 3:
        return None
    
    # Detrend and filter
    detrended = detrend_signal(signal)
    filtered = bandpass_filter(detrended, fps)
    
    # ========== Time Domain Features ==========
    features['std'] = np.std(filtered)
    features['mean_abs'] = np.mean(np.abs(filtered))
    features['rms'] = np.sqrt(np.mean(filtered**2))
    features['skewness'] = skew(filtered)
    features['kurtosis'] = kurtosis(filtered)
    
    # Peak detection
    peaks, properties = find_peaks(filtered, distance=int(fps*0.5), prominence=0.1*np.std(filtered))
    features['num_peaks'] = len(peaks)
    
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / fps
        features['peak_interval_mean'] = np.mean(peak_intervals)
        features['peak_interval_std'] = np.std(peak_intervals)
        features['peak_regularity'] = 1.0 / (1.0 + features['peak_interval_std'])
    else:
        features['peak_interval_mean'] = 0
        features['peak_interval_std'] = 0
        features['peak_regularity'] = 0
    
    # ========== Frequency Domain Features ==========
    
    # FFT analysis
    N = len(filtered)
    fft_vals = np.abs(fft(filtered))
    freqs = fftfreq(N, d=1/fps)
    
    # Focus on physiological range (0.7-4.0 Hz = 42-240 BPM)
    hr_mask = (freqs >= 0.7) & (freqs <= 4.0)
    
    if np.any(hr_mask):
        hr_freqs = freqs[hr_mask]
        hr_power = fft_vals[hr_mask]
        
        # Dominant frequency
        peak_idx = np.argmax(hr_power)
        features['dominant_freq'] = hr_freqs[peak_idx]
        features['heart_rate_bpm'] = hr_freqs[peak_idx] * 60
        features['peak_power'] = hr_power[peak_idx]
        
        # Power concentration
        total_power = np.sum(fft_vals)
        hr_power_sum = np.sum(hr_power)
        features['hr_power_ratio'] = hr_power_sum / (total_power + 1e-10)
        
        # Spectral purity (how concentrated is the power around peak)
        top_3_power = np.sum(np.sort(hr_power)[-3:])
        features['spectral_purity'] = top_3_power / (hr_power_sum + 1e-10)
        
        # SNR estimation
        signal_power = np.max(hr_power)
        noise_power = np.median(hr_power)
        features['snr'] = signal_power / (noise_power + 1e-10)
        
        # Spectral entropy (regularity measure)
        normalized_power = hr_power / (hr_power_sum + 1e-10)
        entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
        features['spectral_entropy'] = entropy
        
    else:
        features.update({
            'dominant_freq': 0, 'heart_rate_bpm': 0, 'peak_power': 0,
            'hr_power_ratio': 0, 'spectral_purity': 0, 'snr': 0, 'spectral_entropy': 0
        })
    
    # Welch's method for power spectral density (more robust than FFT)
    try:
        f_welch, psd_welch = welch(filtered, fps, nperseg=min(256, len(filtered)))
        hr_mask_welch = (f_welch >= 0.7) & (f_welch <= 4.0)
        
        if np.any(hr_mask_welch):
            features['psd_peak'] = np.max(psd_welch[hr_mask_welch])
            features['psd_mean'] = np.mean(psd_welch[hr_mask_welch])
        else:
            features['psd_peak'] = 0
            features['psd_mean'] = 0
    except:
        features['psd_peak'] = 0
        features['psd_mean'] = 0
    
    return features

# -------------------- Suspicious Segment Detection (Enhanced) --------------------
def detect_suspicious_segments(signal, fps, features):
    """Enhanced suspicious segment detection with multiple criteria"""
    window_sec = 3.0
    step_sec = 1.0
    window = int(window_sec * fps)
    step = int(step_sec * fps)

    suspicious = []
    global_std = np.std(signal)
    
    for start in range(0, len(signal) - window, step):
        segment = signal[start:start + window]
        
        # Multiple detection criteria
        amp = np.max(segment) - np.min(segment)
        seg_std = np.std(segment)
        seg_mean = np.mean(np.abs(segment))
        
        reasons = []
        
        # Low amplitude variation
        if amp < 0.3 * global_std:
            reasons.append("Very low amplitude variation")
        
        # Unusual standard deviation
        if seg_std < 0.2 * global_std or seg_std > 3.0 * global_std:
            reasons.append("Abnormal signal variability")
        
        # Check for flat segments (common in deepfakes)
        diff = np.diff(segment)
        if np.sum(np.abs(diff) < 0.01 * global_std) > 0.7 * len(diff):
            reasons.append("Suspiciously flat signal")
        
        if reasons:
            suspicious.append({
                "start_time_sec": round(start / fps, 2),
                "end_time_sec": round((start + window) / fps, 2),
                "reasons": reasons
            })
    
    return suspicious if suspicious else None

# ==================== FIXED CLASSIFICATION LOGIC ====================
def classify_video(features, motion_penalty, suspicious_segments=None):
    """
    Balanced-Strict Classification Logic:
    - Slightly tighter gates to filter improved deepfakes
    - Requires >1.0 SNR for significant points
    - penalizes ambiguity more than the previous relaxed version
    """
    if features is None:
        return "FAKE", 0.0, "Insufficient signal quality"

    score = 0.0
    max_score = 100.0
    reasons = []

    # --- 1. Gating Criteria (SLIGHTLY STRICTER) ---
    # Reject noise floor signals
    if features['std'] < 0.01:
        return "FAKE", 0.0, f"Signal below noise floor (std={features['std']:.4f})"

    # --- 2. Scoring Criteria ---

    # Criterion 1: Signal Strength (20 pts)
    # Real rPPG > 0.2 (normalized). Weak < 0.1
    if features['std'] > 0.18:
        score += 20
        reasons.append(f"Strong signal: {features['std']:.3f}")
    elif features['std'] > 0.05:  # Raised from 0.04
        # Linear scaling for weak-to-moderate signals
        pts = 8 + int((features['std'] - 0.05) * 80)
        score += min(pts, 15)
        reasons.append(f"Moderate signal: {features['std']:.3f} (+{min(pts, 15)})")
    else:
        # Very weak signal
        score += 2
        reasons.append(f"Weak signal: {features['std']:.3f}")

    # Criterion 2: Signal to Noise Ratio (20 pts)
    # Stricter: Requires signal to be larger than noise
    if features['snr'] > 2.2:
        score += 20
        reasons.append(f"High SNR: {features['snr']:.2f}")
    elif features['snr'] > 1.0:  # Raised from 0.8
        score += 15
        reasons.append(f"Moderate SNR: {features['snr']:.2f}")
    elif features['snr'] > 0.5:
        score += 5
        reasons.append(f"Low SNR: {features['snr']:.2f}")
    else:
        reasons.append(f"Very Low SNR: {features['snr']:.2f}")

    # Criterion 3: Heart Rate Validity (20 pts)
    hr = features['heart_rate_bpm']
    if 50 <= hr <= 135:
        if features['snr'] > 0.8: # Raised from 0.5
            score += 20
            reasons.append(f"Valid HR: {hr:.1f} BPM")
        else:
            score += 8
            reasons.append(f"Valid HR (Noisy): {hr:.1f} BPM")
    elif 40 <= hr <= 180:
        score += 5
        reasons.append(f"Borderline HR: {hr:.1f} BPM")
    else:
        reasons.append(f"Invalid HR: {hr:.1f} BPM")

    # Criterion 4: Spectral Concentration (10 pts)
    if features['hr_power_ratio'] > 0.28:
        score += 10
        reasons.append(f"Focused power: {features['hr_power_ratio']:.2f}")
    elif features['hr_power_ratio'] > 0.18:
        score += 5

    # Criterion 5: Peak Regularity (15 pts)
    if features['peak_regularity'] > 0.45:
        score += 15
        reasons.append(f"Regular rhythm: {features['peak_regularity']:.2f}")
    elif features['peak_regularity'] > 0.28:
        score += 8
        reasons.append(f"Semi-regular: {features['peak_regularity']:.2f}")

    # Criterion 6: Spectral Purity (15 pts)
    if features['spectral_purity'] > 0.4:
        score += 15
        reasons.append(f"Pure spectrum: {features['spectral_purity']:.2f}")
    elif features['spectral_purity'] > 0.25:
        score += 10
        reasons.append(f"Moderate purity: {features['spectral_purity']:.2f}")

    # --- 3. Penalties ---

    # Motion Penalty
    if motion_penalty > 0.8:
        deduction = min(motion_penalty * 8, 20) # Slightly higher cap
        score -= deduction
        reasons.append(f"Motion penalty: -{deduction:.1f}")

    # Suspicious Segments Penalty
    if suspicious_segments:
        num_segs = len(suspicious_segments)
        deduction = min(num_segs * 12, 40) # Higher penalty per segment
        score -= deduction
        reasons.append(f"Suspicious segments: -{deduction} ({num_segs})")

    # --- 4. Verdict ---
    confidence = max(0.0, min(score / 95.0, 1.0)) 
    
    # 0.45 is the "Slightly Strict" cutoff
    THRESHOLD = 0.45
    verdict = "REAL" if confidence >= THRESHOLD else "FAKE"
    
    reason_text = "\n".join(reasons)
    
    return verdict, confidence, reason_text

# -------------------- Main rPPG Extraction (Enhanced) --------------------
def extract_rppg(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        fps = 30  # Default fallback

    # Signal storage for multiple methods
    forehead_rgb = []
    cheek_rgb = []
    green_signals = []  # Backup simple method
    
    total_frames = 0
    used_frames = 0
    prev_gray = None
    motion_scores = []
    
    # Enhanced landmark indices
    FOREHEAD = [10, 67, 69, 104, 108, 151, 337, 338]
    LEFT_CHEEK = [234, 93, 132, 58, 172, 136]
    RIGHT_CHEEK = [454, 323, 361, 288, 397, 365]

    print(f"Processing video at {fps} FPS...")
    
    max_duration_sec = 15.0  # Limit processing to first 15 seconds
    max_frames = int(max_duration_sec * fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        
        # Optimization: Stop after max_duration
        if total_frames > max_frames:
            print(f"Reached max duration of {max_duration_sec}s, stopping analysis.")
            break

        # Optimization: Downscale for expensive operations
        h, w = frame.shape[:2]
        scale_factor = 640 / w if w > 640 else 1.0
        
        if scale_factor < 1.0:
            small_w = int(w * scale_factor)
            small_h = int(h * scale_factor)
            frame_small = cv2.resize(frame, (small_w, small_h))
        else:
            frame_small = frame

        # Motion analysis
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray_motion = cv2.resize(gray_small, (320, int(320*h/w))) if w > 320 else gray_small
        
        if prev_gray is not None:
            if prev_gray.shape == gray_motion.shape:
                motion_scores.append(calculate_motion_robustly(prev_gray, gray_motion))
        prev_gray = gray_motion

        # Face detection
        rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb_small)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract ROIs from original high-res frame
        roi_f = get_roi_from_landmarks(frame, landmarks, FOREHEAD)
        roi_l = get_roi_from_landmarks(frame, landmarks, LEFT_CHEEK)
        roi_r = get_roi_from_landmarks(frame, landmarks, RIGHT_CHEEK)

        # Extract RGB values
        rgb_f = extract_rgb_from_roi(roi_f)
        rgb_l = extract_rgb_from_roi(roi_l)
        rgb_r = extract_rgb_from_roi(roi_r)
        
        if rgb_f is None or rgb_l is None or rgb_r is None:
            continue

        forehead_rgb.append(rgb_f)
        cheek_rgb.append((rgb_l + rgb_r) / 2)
        
        # Backup: simple green channel
        if roi_f is not None and roi_f.size > 0:
            green_signals.append(np.mean(roi_f[:, :, 1]))
        
        used_frames += 1

    cap.release()

    # Validate sufficient frames
    min_frames = int(fps * 5)  # Minimum 5 seconds
    if used_frames < min_frames:
        return None, {
            "error": f"Insufficient stable frames (got {used_frames}, need {min_frames})",
            "fps": fps,
            "total_frames": total_frames,
            "used_frames": used_frames
        }

    print(f"Extracted {used_frames}/{total_frames} usable frames")

    # Calculate motion penalty
    motion_penalty = np.mean(motion_scores) if motion_scores else 0
    
    # ========== Method 1: POS Method (Primary) ==========
    forehead_pos = apply_pos_method(forehead_rgb)
    cheek_pos = apply_pos_method(cheek_rgb)
    
    if forehead_pos is not None and cheek_pos is not None:
        fused_pos = 0.6 * forehead_pos + 0.4 * cheek_pos
        fused_pos = detrend_signal(fused_pos)
        filtered_pos = bandpass_filter(fused_pos, fps)
        primary_signal = filtered_pos
        method_used = "POS"
    else:
        primary_signal = None
    
    # ========== Method 2: CHROM Method (Secondary) ==========
    if primary_signal is None:
        forehead_chrom = apply_chrom_method(forehead_rgb, fps)
        cheek_chrom = apply_chrom_method(cheek_rgb, fps)
        
        if forehead_chrom is not None and cheek_chrom is not None:
            fused_chrom = 0.6 * forehead_chrom + 0.4 * cheek_chrom
            filtered_chrom = bandpass_filter(fused_chrom, fps)
            primary_signal = filtered_chrom
            method_used = "CHROM"
    
    # ========== Method 3: Simple Green Channel (Fallback) ==========
    if primary_signal is None and len(green_signals) > 0:
        green_array = np.array(green_signals)
        green_detrended = detrend_signal(green_array)
        primary_signal = bandpass_filter(green_detrended, fps)
        method_used = "Green Channel"
    
    if primary_signal is None:
        return None, {
            "error": "Failed to extract rPPG signal using all methods",
            "fps": fps,
            "total_frames": total_frames,
            "used_frames": used_frames
        }
    
    print(f"Using {method_used} method")
    
    # ========== Extract Advanced Features ==========
    features = extract_advanced_features(primary_signal, fps)
    
    if features is None:
        return None, {
            "error": "Feature extraction failed",
            "fps": fps,
            "total_frames": total_frames,
            "used_frames": used_frames
        }
    
    # ========== Suspicious Segments ==========
    suspicious_segments = detect_suspicious_segments(primary_signal, fps, features)
    
    # ========== Classification ==========
    verdict, confidence, reason_text = classify_video(features, motion_penalty, suspicious_segments)
    
    # ========== Generate Plots ==========
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: rPPG Waveform
    plt.figure(figsize=(12, 4))
    time_axis = np.arange(len(primary_signal)) / fps
    plt.plot(time_axis, primary_signal, linewidth=0.8)
    plt.title(f"rPPG Signal ({method_used} Method)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True, alpha=0.3)
    waveform_path = "plots/rppg_waveform.png"
    plt.savefig(waveform_path, dpi=150)
    plt.close()
    
    # Plot 2: FFT Spectrum
    N = len(primary_signal)
    fft_vals = np.abs(fft(primary_signal))
    freqs = fftfreq(N, d=1/fps)
    
    plt.figure(figsize=(12, 4))
    hr_mask = (freqs >= 0) & (freqs <= 5)
    plt.plot(freqs[hr_mask], fft_vals[hr_mask], linewidth=1.5)
    plt.axvline(x=features['dominant_freq'], color='r', linestyle='--', 
                label=f'Dominant: {features["heart_rate_bpm"]:.1f} BPM')
    plt.fill_between([0.7, 4.0], 0, plt.ylim()[1], alpha=0.2, color='green', 
                     label='Physiological Range')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fft_path = "plots/fft_spectrum.png"
    plt.savefig(fft_path, dpi=150)
    plt.close()
    
    # Compile results
    result = {
        "verdict": verdict,
        "confidence": float(confidence),
        "reason": reason_text,
        "method_used": method_used,
        "features": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                     for k, v in features.items()},
        "fps": float(fps),
        "total_frames": int(total_frames),
        "used_frames": int(used_frames),
        "motion_penalty": float(motion_penalty),
        "waveform_plot": waveform_path,
        "fft_plot": fft_path,
        "suspicious_segments": suspicious_segments
    }
    
    return result, None