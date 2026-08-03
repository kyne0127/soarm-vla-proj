import argparse
import csv
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.gesture.gesture_recognizer import GestureRecognizer
from modules.sensors.zed_camera import ZEDCameraManager


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep ablation for RANSAC plane fitting")
    parser.add_argument("--seconds", type=int, default=8, help="Capture duration per config")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hand-model-path", type=str, default="models/hand_landmarker.task")
    parser.add_argument("--camera-to-robot-path", type=str, default="data/calibration/camera_to_robot.json")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.01,0.02,0.03",
        help="Comma-separated RANSAC distance thresholds in meters",
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default="60,120,180",
        help="Comma-separated RANSAC iteration counts",
    )
    return parser.parse_args()


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _jitter(points):
    if len(points) < 2:
        return None
    diffs = [_dist(points[i], points[i - 1]) for i in range(1, len(points))]
    return float(np.mean(diffs))


def _spread(points):
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    center = np.mean(arr, axis=0)
    d = np.linalg.norm(arr - center, axis=1)
    return float(np.mean(d))


def evaluate(log):
    frames_total = log["frames_total"]
    target_points = log["target_points"]
    elapsed = max(log["elapsed_sec"], 1e-6)

    return {
        "frames_total": frames_total,
        "pointing_frames": log["pointing_frames"],
        "target_detected_frames": len(target_points),
        "target_detected_ratio": round(len(target_points) / frames_total, 4) if frames_total > 0 else 0.0,
        "effective_fps": round(frames_total / elapsed, 3),
        "mean_inference_ms": round(float(np.mean(log["infer_ms"])) if log["infer_ms"] else 0.0, 3),
        "target_jitter_m": round(_jitter(target_points), 5) if _jitter(target_points) is not None else None,
        "target_spread_m": round(_spread(target_points), 5) if _spread(target_points) is not None else None,
    }


def run_single_config(seconds, recognizer, zed):
    log = {
        "frames_total": 0,
        "pointing_frames": 0,
        "target_points": [],
        "infer_ms": [],
        "elapsed_sec": 0.0,
    }

    start = time.time()
    end = start + seconds
    while time.time() < end:
        rgb, _, point_cloud = zed.get_frames()
        if rgb is None or point_cloud is None:
            continue

        t0 = time.perf_counter()
        state = recognizer.recognize(rgb, point_cloud)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        log["frames_total"] += 1
        log["infer_ms"].append(infer_ms)
        if state.get("label") == "POINTING":
            log["pointing_frames"] += 1
        target = state.get("pointing_target_3d")
        if target is not None:
            log["target_points"].append(target)

    log["elapsed_sec"] = time.time() - start
    return evaluate(log)


def main():
    args = parse_args()
    thresholds = parse_float_list(args.thresholds)
    iterations = parse_int_list(args.iterations)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / "data" / "test_runs" / f"ablation_sweep_{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    zed = ZEDCameraManager(fps=args.fps)
    zed.start()

    results = []
    try:
        baseline_rec = GestureRecognizer(
            hand_model_path=args.hand_model_path,
            camera_to_robot_path=args.camera_to_robot_path,
            plane_method="z_median",
        )
        baseline_metrics = run_single_config(args.seconds, baseline_rec, zed)
        baseline_rec.close()
        baseline_row = {
            "plane_method": "z_median",
            "ransac_threshold_m": None,
            "ransac_iterations": None,
            **baseline_metrics,
        }
        results.append(baseline_row)
        print(
            f"[BASELINE] z_median -> ratio={baseline_metrics['target_detected_ratio']}, "
            f"jitter={baseline_metrics['target_jitter_m']}, infer_ms={baseline_metrics['mean_inference_ms']}"
        )

        for th, it in product(thresholds, iterations):
            rec = GestureRecognizer(
                hand_model_path=args.hand_model_path,
                camera_to_robot_path=args.camera_to_robot_path,
                plane_method="ransac",
                ransac_iterations=it,
                ransac_threshold_m=th,
            )
            metrics = run_single_config(args.seconds, rec, zed)
            rec.close()

            row = {
                "plane_method": "ransac",
                "ransac_threshold_m": th,
                "ransac_iterations": it,
                **metrics,
            }
            results.append(row)
            print(f"[CONFIG] th={th:.3f}, it={it} -> ratio={metrics['target_detected_ratio']}, jitter={metrics['target_jitter_m']}, infer_ms={metrics['mean_inference_ms']}")
    finally:
        zed.stop()

    # Higher detect ratio, lower jitter, lower latency are preferred.
    def score(item):
        ratio = item["target_detected_ratio"]
        jitter = item["target_jitter_m"] if item["target_jitter_m"] is not None else 1.0
        infer_ms = item["mean_inference_ms"]
        return ratio - 0.5 * jitter - 0.002 * infer_ms

    ranked = sorted(results, key=score, reverse=True)
    best_ransac = next((item for item in ranked if item["plane_method"] == "ransac"), None)
    baseline = next((item for item in results if item["plane_method"] == "z_median"), None)

    report = {
        "config": {
            "seconds_per_config": args.seconds,
            "fps": args.fps,
            "thresholds": thresholds,
            "iterations": iterations,
        },
        "results": results,
        "ranked_top3": ranked[:3],
        "best_ransac": best_ransac,
        "baseline_z_median": baseline,
    }

    if best_ransac is not None and baseline is not None:
        report["best_ransac_vs_baseline"] = {
            "target_detected_ratio": round(best_ransac["target_detected_ratio"] - baseline["target_detected_ratio"], 4),
            "target_jitter_m": (
                round(best_ransac["target_jitter_m"] - baseline["target_jitter_m"], 5)
                if best_ransac["target_jitter_m"] is not None and baseline["target_jitter_m"] is not None
                else None
            ),
            "mean_inference_ms": round(best_ransac["mean_inference_ms"] - baseline["mean_inference_ms"], 3),
        }

    json_path = out_root / "ablation_sweep_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "ablation_sweep_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "plane_method",
            "ransac_threshold_m",
            "ransac_iterations",
            "frames_total",
            "pointing_frames",
            "target_detected_frames",
            "target_detected_ratio",
            "effective_fps",
            "mean_inference_ms",
            "target_jitter_m",
            "target_spread_m",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("[DONE] Sweep ablation report generated")
    print(f" - json: {json_path}")
    print(f" - csv:  {csv_path}")
    if ranked:
        best = ranked[0]
        print(
            f" - best: method={best['plane_method']}, th={best['ransac_threshold_m']}, "
            f"it={best['ransac_iterations']}, ratio={best['target_detected_ratio']}, "
            f"jitter={best['target_jitter_m']}, infer_ms={best['mean_inference_ms']}"
        )


if __name__ == "__main__":
    main()
