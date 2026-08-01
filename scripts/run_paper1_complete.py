"""
Paper 1 Complete Summary Generator
"""
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/sentinel/sentinel_iot/paper1/results")

def load_json(path):
    try:
        with open(str(path)) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def run():
    print("="*60)
    print("PAPER 1 — RCAF-IoT COMPLETE SUMMARY")
    print("Target: IEEE Internet of Things Journal (IF ~10.6)")
    print("="*60)

    # Load all results
    stride  = load_json(RESULTS_DIR / "stride_dread_threat_model.json")
    rcaf    = load_json(RESULTS_DIR / "rcaf_evaluation_results.json")
    privacy = load_json(RESULTS_DIR / "privacy_module_evaluation.json")

    # STRIDE summary
    print("\n[1] STRIDE-DREAD Threat Model")
    stats = stride.get("statistics", {})
    print(f"    Total threats:   {stats.get('total_threats', 'N/A')}")
    print(f"    Critical:        {stats.get('critical', 'N/A')}")
    print(f"    High:            {stats.get('high', 'N/A')}")
    print(f"    Avg DREAD score: {stats.get('average_dread_score', 'N/A')}")

    # RCAF summary
    print("\n[2] RCAF Engine Results")
    metrics = rcaf.get("evaluation_metrics", {})
    bench   = rcaf.get("hardware_benchmark", {})
    lat     = bench.get("single_sample_latency_ms", {})
    print(f"    F1 Score:        {metrics.get('f1_score', 'N/A'):.4f}")
    print(f"    Detection Rate:  {metrics.get('detection_rate', 'N/A'):.4f}")
    print(f"    False Pos Rate:  {metrics.get('false_positive_rate', 'N/A'):.4f}")
    print(f"    ROC-AUC:         {metrics.get('roc_auc', 'N/A'):.4f}")
    print(f"    Avg Latency:     {metrics.get('avg_latency_ms_per_sample', lat.get('mean', 'N/A')):.3f} ms")

    # Privacy summary
    print("\n[3] Privacy Module Results")
    tradeoff = privacy.get("privacy_accuracy_tradeoff", {})
    if "1.0" in tradeoff:
        r = tradeoff["1.0"]
        print(f"    epsilon=1.0 PSNR:        {r.get('mean_psnr_db', 'N/A'):.2f} dB")
        print(f"    Occupancy preservation:  {r.get('mean_occupancy_preservation', 'N/A'):.4f}")
        print(f"    Processing latency:      {r.get('mean_processing_latency_ms', 'N/A'):.2f} ms")
    if "0.1" in tradeoff:
        r2 = tradeoff["0.1"]
        print(f"    epsilon=0.1 PSNR:        {r2.get('mean_psnr_db', 'N/A'):.2f} dB (max privacy)")

    # Save complete summary
    summary = {
        "paper": "Paper 1: RCAF-IoT",
        "title": "RCAF-IoT: A Lightweight RGB-Depth Context-Aware "
                 "Anomaly Fusion Engine for Resource-Constrained IoT Edge Nodes",
        "target_journal": "IEEE Internet of Things Journal",
        "issn": "2327-4662",
        "impact_factor": "~10.6",
        "generated": datetime.now().isoformat(),
        "hardware": {
            "edge_node": "Raspberry Pi 4B ARM Cortex-A72 @ 1.8GHz",
            "sensor": "Microsoft Kinect Xbox 360 Model 1473 (Kinect v1)"
        },
        "contributions": {
            "C1": "First STRIDE-DREAD threat model for trimodal Kinect v1 + Pi 4",
            "C2": "RCAF: 48-dim depth-weighted fusion anomaly detection on ARM",
            "C3": "Depth-Silhouette Privacy Module with epsilon-PSNR tradeoff",
            "C4": "ARM-validated inference benchmarks on Pi 4"
        },
        "key_results": {
            "stride_threats": stats.get("total_threats"),
            "critical_threats": stats.get("critical"),
            "avg_dread_score": stats.get("average_dread_score"),
            "f1_score": metrics.get("f1_score"),
            "detection_rate": metrics.get("detection_rate"),
            "false_positive_rate": metrics.get("false_positive_rate"),
            "avg_latency_ms": metrics.get("avg_latency_ms_per_sample"),
            "privacy_psnr_eps1": tradeoff.get("1.0", {}).get("mean_psnr_db"),
            "privacy_occupancy_eps1": tradeoff.get("1.0", {}).get(
                "mean_occupancy_preservation")
        },
        "result_files": [
            "stride_dread_threat_model.json",
            "stride_dread_threat_model.csv",
            "rcaf_evaluation_results.json",
            "privacy_module_evaluation.json"
        ]
    }

    out = RESULTS_DIR / "paper1_complete_summary.json"
    with open(str(out), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("PAPER 1 COMPLETE")
    print("="*60)
    print(f"All results in: {RESULTS_DIR}")
    print(f"Summary saved:  {out}")
    print("\nNext: Run Paper 2 pipeline")
    print("="*60)

if __name__ == "__main__":
    run()
