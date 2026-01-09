
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks, hilbert, detrend
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Initialize face detection
mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.6
)

# ============= 1. SKIN SEGMENTATION =============
def extract_skin_pixels(roi):
    if roi is None or roi.size == 0:
        return None
    
    # YCrCb color space for robust skin detection
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # Clean mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Check if enough skin pixels exist
    if np.sum(mask > 0) < 0.3 * mask.size:
        return None
    
    # Return mean RGB of skin pixels only
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    skin_pixels = roi_rgb[mask > 0]
    return np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else None

# ============= 2. SIGNAL FILTERING =============
def bandpass_filter(signal, fps, low=0.7, high=4.0):
    nyquist = fps / 2.0
    b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
    return filtfilt(b, a, signal)

# ============= 3. POS METHOD =============
def extract_pulse_pos(rgb_signals):
    rgb_signals = np.array(rgb_signals)
    if len(rgb_signals) < 10:
        return None
    
    # Normalize RGB channels
    mean_rgb = np.mean(rgb_signals, axis=0)
    if np.any(mean_rgb == 0):
        return None
    normalized = rgb_signals / mean_rgb
    
    # Project onto plane orthogonal to skin tone
    S = np.array([[0, 1, -1], [-2, 1, 1]])
    P = np.dot(S, normalized.T)
    
    # Extract pulse with adaptive weight
    alpha = np.std(P[0]) / (np.std(P[1]) + 1e-10)
    pulse = P[0] - alpha * P[1]
    return pulse

# ============= 4. CROSS-ROI COHERENCE =============
def compute_coherence(signal1, signal2):
    if len(signal1) != len(signal2) or len(signal1) < 30:
        return None
    
    # Correlation coefficient
    correlation, _ = pearsonr(signal1, signal2)
    
    # Phase synchronization using Hilbert transform
    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))
    phase_diff = phase1 - phase2
    phase_sync = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return {
        'correlation': float(correlation),
        'phase_sync': float(phase_sync),
        'coherent': bool(correlation > 0.6)  # Threshold for real videos
    }

# ============= 5. SLIDING WINDOW ANALYSIS =============
def analyze_windows(signal, fps, window_sec=4):
    window_size = int(window_sec * fps)
    step = window_size // 2  # 50% overlap
    
    # For very short videos, use smaller windows
    if len(signal) < window_size:
        window_sec = max(2, len(signal) // (2 * fps))  # Minimum 2 seconds
        window_size = int(window_sec * fps)
        step = window_size // 2
    
    if len(signal) < window_size:
        return None
    
    window_verdicts = []
    window_features = []
    
    # Analyze each window
    for start in range(0, len(signal) - window_size, step):
        window = signal[start:start + window_size]
        features = extract_features(window, fps)
        
        if features:
            window_features.append(features)
            # Simple classification: good HR + SNR = REAL
            is_real = (50 <= features['hr'] <= 120 and features['snr'] > 2.0)
            window_verdicts.append('REAL' if is_real else 'FAKE')
    
    if not window_verdicts:
        return None
    
    # Majority voting
    real_count = window_verdicts.count('REAL')
    
    # Check temporal consistency
    hrs = [f['hr'] for f in window_features]
    hr_consistency = 1.0 / (1.0 + np.std(hrs))
    
    return {
        'total_windows': len(window_verdicts),
        'real_windows': real_count,
        'fake_windows': len(window_verdicts) - real_count,
        'consensus': 'REAL' if real_count > len(window_verdicts)/2 else 'FAKE',
        'hr_consistency': float(hr_consistency),
        'features': window_features
    }

# ============= 6. FEATURE EXTRACTION =============
def extract_features(signal, fps):
    # Preprocess
    signal = detrend(signal)
    signal = bandpass_filter(signal, fps)
    
    # FFT for frequency analysis
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(len(signal), 1/fps)
    
    # Focus on heart rate range (0.7-4.0 Hz = 42-240 BPM)
    hr_mask = (freqs >= 0.7) & (freqs <= 4.0)
    hr_power = fft_vals[hr_mask]
    hr_freqs = freqs[hr_mask]
    
    if len(hr_power) == 0:
        return None
    
    # Find dominant frequency (heart rate)
    peak_idx = np.argmax(hr_power)
    heart_rate = hr_freqs[peak_idx] * 60  # Convert to BPM
    
    # Signal quality metrics
    snr = np.max(hr_power) / (np.median(hr_power) + 1e-10)
    power_ratio = np.sum(hr_power) / (np.sum(fft_vals) + 1e-10)
    
    # Peak regularity
    peaks, _ = find_peaks(signal, distance=int(fps*0.5))
    regularity = 0.0
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fps
        regularity = 1.0 / (1.0 + np.std(intervals))
    
    return {
        'hr': float(heart_rate),
        'snr': float(snr),
        'power_ratio': float(power_ratio),
        'regularity': float(regularity),
        'signal_strength': float(np.std(signal))
    }

# ============= 7. PHYSIOLOGICAL RULES =============
def check_physiology(features, coherence):
    rules = {
        'valid_hr': bool(42 <= features['hr'] <= 180),
        'good_snr': bool(features['snr'] > 1.5),
        'concentrated_spectrum': bool(features['power_ratio'] > 0.25),
        'regular_peaks': bool(features['regularity'] > 0.5),
        'cross_roi_sync': bool(coherence and coherence['coherent'])
    }
    
    passed = sum(rules.values())
    return {
        'rules': rules,
        'passed': int(passed),
        'total': int(len(rules)),
        'score': float(passed / len(rules))
    }

# ============= 8. SUSPICIOUS SEGMENTS =============
def find_suspicious_segments(signal, fps, window_features):
    """Find specific time segments that look fake"""
    segments = []
    
    # Adapt window size for shorter videos
    window_sec = min(4, len(signal) / fps / 2)  # Use smaller windows for short videos
    step_sec = window_sec / 2
    
    # Check each window
    for i, features in enumerate(window_features):
        start_time = i * step_sec
        end_time = start_time + window_sec
        
        reasons = []
        
        # Check multiple criteria
        if features['hr'] < 42 or features['hr'] > 180:
            reasons.append("Invalid heart rate")
        if features['snr'] < 1.5:
            reasons.append("Low signal quality")
        if features['regularity'] < 0.4:
            reasons.append("Irregular heartbeat")
        
        if reasons:
            segments.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'reasons': reasons
            })
    
    # Also check for flat regions in full signal
    check_window = min(int(2 * fps), len(signal) // 3)  # Adaptive window size
    global_std = np.std(signal)
    
    for start in range(0, len(signal) - check_window, int(fps)):
        segment = signal[start:start + check_window]
        if np.max(segment) - np.min(segment) < 0.3 * global_std:
            start_time = round(start / fps, 2)
            end_time = round((start + check_window) / fps, 2)
            
            # Avoid duplicates
            if not any(abs(s['start'] - start_time) < 2 for s in segments):
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'reasons': ['Low amplitude variation']
                })
    
    return sorted(segments, key=lambda x: x['start']) if segments else None

# ============= 9. VISUALIZATION =============
def create_plots(signal, fps, features):
    """Generate plots and save to plots folder"""
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: Signal waveform
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(signal)) / fps
    ax.plot(time, signal, linewidth=0.8, color='#2E86AB')
    ax.set_title('rPPG Signal', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    
    plt.savefig('plots/rppg_waveform.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Frequency spectrum
    fig, ax = plt.subplots(figsize=(12, 4))
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(len(signal), 1/fps)
    mask = (freqs >= 0) & (freqs <= 5)
    
    ax.plot(freqs[mask], fft_vals[mask], linewidth=1.5, color='#A23B72')
    ax.axvline(features['hr']/60, color='red', linestyle='--', 
               label=f"HR: {features['hr']:.1f} BPM")
    ax.fill_between([0.7, 4.0], 0, max(fft_vals[mask]), 
                     alpha=0.2, color='green', label='Physiological range')
    ax.set_title('Frequency Spectrum', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.savefig('plots/fft_spectrum.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    return {
        'waveform_saved': 'plots/rppg_waveform.png',
        'spectrum_saved': 'plots/fft_spectrum.png'
    }

# ============= 10. MAIN FUNCTION =============
def extract_rppg(video_path):
    """Main extraction and analysis pipeline"""
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Storage for signals
    forehead_rgb = []
    cheek_rgb = []
    total_frames = 0
    used_frames = 0
    
    # Face landmark indices
    FOREHEAD = [10, 67, 69, 104, 108, 151, 337, 338]
    LEFT_CHEEK = [234, 93, 132, 58, 172]
    RIGHT_CHEEK = [454, 323, 361, 288, 397]
    
    print(f"Processing video at {fps} FPS...")
    
    # Extract signals from video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        
        # Detect face
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        
        if not results.multi_face_landmarks:
            continue
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Extract ROIs
        def get_roi(indices):
            points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                     for i in indices if i < len(landmarks)]
            if len(points) < 3:
                return None
            xs, ys = zip(*points)
            x1, x2 = max(min(xs), 0), min(max(xs), w)
            y1, y2 = max(min(ys), 0), min(max(ys), h)
            return frame[y1:y2, x1:x2] if x2-x1 > 10 and y2-y1 > 10 else None
        
        roi_f = get_roi(FOREHEAD)
        roi_l = get_roi(LEFT_CHEEK)
        roi_r = get_roi(RIGHT_CHEEK)
        
        # Try skin segmentation first
        rgb_f = extract_skin_pixels(roi_f)
        rgb_l = extract_skin_pixels(roi_l)
        rgb_r = extract_skin_pixels(roi_r)
        
        # Fallback: if skin segmentation fails, use simple mean
        if rgb_f is None and roi_f is not None:
            rgb_f = np.mean(cv2.cvtColor(roi_f, cv2.COLOR_BGR2RGB).reshape(-1, 3), axis=0)
        if rgb_l is None and roi_l is not None:
            rgb_l = np.mean(cv2.cvtColor(roi_l, cv2.COLOR_BGR2RGB).reshape(-1, 3), axis=0)
        if rgb_r is None and roi_r is not None:
            rgb_r = np.mean(cv2.cvtColor(roi_r, cv2.COLOR_BGR2RGB).reshape(-1, 3), axis=0)
        
        if rgb_f is not None and rgb_l is not None and rgb_r is not None:
            forehead_rgb.append(rgb_f)
            cheek_rgb.append((rgb_l + rgb_r) / 2)
            used_frames += 1
    
    cap.release()
    
    # Check if enough frames (adaptive minimum based on video length)
    min_required = max(int(fps * 2), 60)  # At least 2 seconds OR 60 frames
    frame_quality = (used_frames / total_frames * 100) if total_frames > 0 else 0
    
    if used_frames < min_required:
        return None, {
            "error": f"Insufficient quality frames: {used_frames}/{total_frames} frames usable ({frame_quality:.1f}% pass rate)",
            "suggestion": "Ensure: (1) Face clearly visible, (2) Good lighting, (3) Minimal motion, (4) Camera focused on face",
            "fps": fps,
            "total_frames": total_frames,
            "used_frames": used_frames,
            "min_required": min_required
        }
    
    print(f"Extracted {used_frames}/{total_frames} frames ({frame_quality:.1f}% quality)")
    
    # Extract pulse signals using POS method
    forehead_pulse = extract_pulse_pos(forehead_rgb)
    cheek_pulse = extract_pulse_pos(cheek_rgb)
    
    if forehead_pulse is None or cheek_pulse is None:
        return None, {"error": "Failed to extract pulse signals"}
    
    # Preprocess signals
    forehead_pulse = bandpass_filter(detrend(forehead_pulse), fps)
    cheek_pulse = bandpass_filter(detrend(cheek_pulse), fps)
    
    # Fuse signals (weighted average)
    fused_signal = 0.6 * forehead_pulse + 0.4 * cheek_pulse
    
    # === ANALYSIS ===
    
    # 1. Cross-ROI coherence
    print("Computing cross-ROI coherence...")
    coherence = compute_coherence(forehead_pulse, cheek_pulse)
    
    # 2. Sliding window temporal analysis
    print("Performing sliding-window analysis...")
    window_analysis = analyze_windows(fused_signal, fps)
    
    if window_analysis is None:
        return None, {"error": "Window analysis failed"}
    
    # 3. Global features
    global_features = extract_features(fused_signal, fps)
    
    if global_features is None:
        return None, {"error": "Feature extraction failed"}
    
    # 4. Physiological plausibility
    plausibility = check_physiology(global_features, coherence)
    
    # 5. Find suspicious segments
    suspicious = find_suspicious_segments(
        fused_signal, fps, window_analysis['features']
    )
    
    # === FINAL CLASSIFICATION ===
    
    score = 0
    
    # Window consensus (40 points)
    if window_analysis['consensus'] == 'REAL':
        score += 40 * (window_analysis['real_windows'] / window_analysis['total_windows'])
    
    # Temporal consistency (20 points)
    score += 20 * window_analysis['hr_consistency']
    
    # Physiological rules (25 points)
    score += 25 * plausibility['score']
    
    # Cross-ROI coherence (15 points)
    if coherence and coherence['coherent']:
        score += 15
    
    confidence = float(max(0.0, min(score / 100, 1.0)))
    verdict = "REAL" if confidence >= 0.50 else "FAKE"
    
    # Generate plots (saved to plots folder)
    print("Generating visualizations...")
    plot_info = create_plots(fused_signal, fps, global_features)
    
    # === RETURN RESULTS ===
    
    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "confidence_percentage": f"{confidence * 100:.1f}%",
        
        "video_info": {
            "fps": float(fps),
            "total_frames": int(total_frames),
            "used_frames": int(used_frames),
            "duration_sec": round(used_frames / fps, 2)
        },
        
        "physiological_metrics": {
            "heart_rate_bpm": float(global_features['hr']),
            "signal_strength": float(global_features['signal_strength']),
            "snr": float(global_features['snr']),
            "peak_regularity": float(global_features['regularity']),
            "spectral_concentration": float(global_features['power_ratio'])
        },
        
        "temporal_analysis": {
            "total_windows": window_analysis['total_windows'],
            "real_windows": window_analysis['real_windows'],
            "fake_windows": window_analysis['fake_windows'],
            "consensus": window_analysis['consensus'],
            "hr_consistency": window_analysis['hr_consistency']
        },
        
        "cross_roi_coherence": coherence,
        
        "physiological_rules": plausibility,
        
        "suspicious_segments": suspicious,
        
        "plots_saved_to": plot_info,
        
        "method": "POS + Skin Segmentation + Sliding Window + Cross-ROI Coherence"
        
    }, None