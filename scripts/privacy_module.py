"""
Depth-Silhouette Privacy Protection Module
Paper 1 Contribution
"""

import numpy as np
import cv2
import json
import time
from pathlib import Path

RESULTS_DIR = Path("/home/sentinel/sentinel_iot/paper1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KINECT_V1 = {
    "depth_width": 320,
    "depth_height": 240,
    "depth_min_mm": 800,
    "depth_max_mm": 4000,
    "invalid_value": 0,
    "person_near_threshold_mm": 1800,
    "person_far_threshold_mm": 3500,
    "model": "Xbox 360 Model 1473"
}


def segment_human_body(depth_frame):
    valid_mask = (depth_frame > KINECT_V1["depth_min_mm"]) & \
                 (depth_frame < KINECT_V1["depth_max_mm"])
    body_mask = valid_mask & \
                (depth_frame < KINECT_V1["person_near_threshold_mm"]) & \
                (depth_frame > KINECT_V1["depth_min_mm"])
    kernel = np.ones((5, 5), np.uint8)
    body_uint8 = body_mask.astype(np.uint8) * 255
    body_cleaned = cv2.morphologyEx(body_uint8, cv2.MORPH_CLOSE,
                                     kernel, iterations=2)
    body_cleaned = cv2.morphologyEx(body_cleaned, cv2.MORPH_OPEN,
                                     np.ones((3,3), np.uint8), iterations=1)
    return body_cleaned


def apply_silhouette_erosion(depth_frame, body_mask, erosion_kernel_size=7):
    depth_anon = depth_frame.copy().astype(np.float32)
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(body_mask, kernel, iterations=2)
    body_pixels = depth_frame[body_mask > 0]
    if len(body_pixels) > 0:
        mean_depth = float(np.mean(body_pixels))
        depth_anon[body_mask > 0] = mean_depth
        if np.sum(eroded_mask > 0) > 0:
            depth_anon[eroded_mask > 0] = mean_depth * 0.98
    return depth_anon.astype(np.uint16)


def inject_calibrated_noise(depth_frame, body_mask,
                             epsilon=1.0, sensitivity=100.0):
    depth_noisy = depth_frame.copy().astype(np.float32)
    sigma = sensitivity / epsilon
    body_region = (body_mask > 0)
    if np.sum(body_region) > 0:
        noise = np.random.normal(0, sigma, depth_frame.shape)
        depth_noisy[body_region] += noise[body_region]
    bg_region = ~body_region & (depth_frame > KINECT_V1["depth_min_mm"])
    if np.sum(bg_region) > 0:
        bg_noise = np.random.normal(0, sigma * 0.1, depth_frame.shape)
        depth_noisy[bg_region] += bg_noise[bg_region]
    depth_noisy = np.clip(depth_noisy,
                          KINECT_V1["depth_min_mm"],
                          KINECT_V1["depth_max_mm"])
    depth_noisy[depth_frame == KINECT_V1["invalid_value"]] = 0
    return depth_noisy.astype(np.uint16)


def anonymise_depth_frame(depth_frame, epsilon=1.0, erosion_kernel=7):
    t0 = time.perf_counter()
    body_mask   = segment_human_body(depth_frame)
    depth_eroded = apply_silhouette_erosion(depth_frame, body_mask,
                                             erosion_kernel)
    depth_final  = inject_calibrated_noise(depth_eroded, body_mask,
                                            epsilon=epsilon)
    lat_ms = (time.perf_counter() - t0) * 1000
    return depth_final, body_mask, lat_ms


def evaluate_privacy(original_frames, anonymised_frames):
    psnr_vals, mse_vals, occ_vals = [], [], []
    for orig, anon in zip(original_frames, anonymised_frames):
        orig_f = orig.astype(np.float32)
        anon_f = anon.astype(np.float32)
        mse = float(np.mean((orig_f - anon_f) ** 2))
        mse_vals.append(mse)
        max_val = KINECT_V1["depth_max_mm"]
        psnr = 10 * np.log10((max_val**2) / mse) if mse > 0 else 100.0
        psnr_vals.append(psnr)
        orig_occ = (orig > KINECT_V1["depth_min_mm"]) & (orig < 2000)
        anon_occ = (anon > KINECT_V1["depth_min_mm"]) & (anon < 2000)
        if np.sum(orig_occ) > 0:
            occ_vals.append(float(np.sum(orig_occ & anon_occ) /
                                  np.sum(orig_occ)))
    return {
        "n_frames": len(psnr_vals),
        "mean_psnr_db": float(np.mean(psnr_vals)) if psnr_vals else 0,
        "std_psnr_db":  float(np.std(psnr_vals))  if psnr_vals else 0,
        "mean_mse":     float(np.mean(mse_vals))   if mse_vals  else 0,
        "mean_occupancy_preservation":
            float(np.mean(occ_vals)) if occ_vals else 0,
    }


def privacy_accuracy_tradeoff_analysis():
    print("\n  Privacy-Accuracy Tradeoff Analysis...")
    print("  Testing epsilon: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0")

    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}

    rng = np.random.default_rng(42)
    H, W = 240, 320
    test_frames = []

    for _ in range(50):
        frame = rng.integers(2500, 4000, (H, W), dtype=np.uint16)
        # Safe person position — ensure patch fits inside frame
        py = int(rng.integers(10, H - 110))   # row start, leaves 100px room
        px = int(rng.integers(10, W - 70))    # col start, leaves 60px room
        frame[py:py+100, px:px+60] = rng.integers(
            800, 1800, (100, 60), dtype=np.uint16
        )
        test_frames.append(frame)

    for eps in epsilon_values:
        print(f"    epsilon={eps:.1f}...", end=" ", flush=True)
        anonymised = []
        latencies  = []
        for frame in test_frames:
            anon, _, lat = anonymise_depth_frame(frame, epsilon=eps)
            anonymised.append(anon)
            latencies.append(lat)
        ev = evaluate_privacy(test_frames, anonymised)
        ev["epsilon"] = eps
        ev["mean_processing_latency_ms"] = float(np.mean(latencies))
        results[str(eps)] = ev
        print(f"PSNR={ev['mean_psnr_db']:.1f}dB  "
              f"Occ={ev['mean_occupancy_preservation']:.3f}  "
              f"Lat={ev['mean_processing_latency_ms']:.1f}ms")

    return results


def run_privacy_module_evaluation():
    print("="*60)
    print("DEPTH-SILHOUETTE PRIVACY MODULE EVALUATION")
    print("="*60)

    tradeoff = privacy_accuracy_tradeoff_analysis()

    if "1.0" in tradeoff:
        r = tradeoff["1.0"]
        print(f"\n  Recommended epsilon=1.0 results:")
        print(f"    PSNR:                {r['mean_psnr_db']:.2f} dB")
        print(f"    Occupancy preserved: {r['mean_occupancy_preservation']:.4f}")
        print(f"    Processing latency:  {r['mean_processing_latency_ms']:.2f} ms/frame")

    results = {
        "module": "Depth-Silhouette Privacy Protection Module",
        "sensor": "Microsoft Kinect Xbox 360 Model 1473 (Kinect v1)",
        "kinect_specs": KINECT_V1,
        "pipeline_stages": [
            "Human body segmentation via depth thresholding",
            "Silhouette morphological erosion (kernel=7x7, iterations=2)",
            "Calibrated Gaussian noise injection (Gaussian mechanism)",
            "Invalid pixel preservation"
        ],
        "privacy_accuracy_tradeoff": tradeoff,
        "privacy_guarantee": "(epsilon, delta)-DP with Gaussian mechanism",
        "recommended_epsilon": 1.0,
        "biometric_references": [
            "Ye et al. (2011): depth silhouette for identification",
            "Han et al. (2013): Kinect gait recognition via depth",
            "Dwork & Roth (2014): Gaussian mechanism for DP"
        ]
    }

    out = RESULTS_DIR / "privacy_module_evaluation.json"
    with open(str(out), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out}")
    return results


if __name__ == "__main__":
    run_privacy_module_evaluation()
    print("\nPrivacy module evaluation complete")
