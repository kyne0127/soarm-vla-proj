import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.gesture.gesture_recognizer import GestureRecognizer
from modules.sensors.zed_camera import ZEDCameraManager


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study: z-median vs RANSAC plane fitting")
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hand-model-path", type=str, default="models/hand_landmarker.task")
    parser.add_argument("--camera-to-robot-path", type=str, default="data/calibration/camera_to_robot.json")
    parser.add_argument("--ransac-iterations", type=int, default=120)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.02)
    return parser.parse_args()


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


def evaluate(run_log):
    frames_total = run_log["frames_total"]
    target_points = run_log["target_points"]
    elapsed = max(run_log["elapsed_sec"], 1e-6)

    return {
        "frames_total": frames_total,
        "pointing_frames": run_log["pointing_frames"],
        "target_detected_frames": len(target_points),
        "target_detected_ratio": round(len(target_points) / frames_total, 4) if frames_total > 0 else 0.0,
        "effective_fps": round(frames_total / elapsed, 3),
        "mean_inference_ms": round(float(np.mean(run_log["infer_ms"])) if run_log["infer_ms"] else 0.0, 3),
        "target_jitter_m": round(_jitter(target_points), 5) if _jitter(target_points) is not None else None,
        "target_spread_m": round(_spread(target_points), 5) if _spread(target_points) is not None else None,
    }


def main():
    args = parse_args()

    out_root = PROJECT_ROOT / "data" / "test_runs" / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    zed = ZEDCameraManager(fps=args.fps)
    rec_z = GestureRecognizer(
        hand_model_path=args.hand_model_path,
        camera_to_robot_path=args.camera_to_robot_path,
        plane_method="z_median",
    )
    rec_r = GestureRecognizer(
        hand_model_path=args.hand_model_path,
        camera_to_robot_path=args.camera_to_robot_path,
        plane_method="ransac",
        ransac_iterations=args.ransac_iterations,
        ransac_threshold_m=args.ransac_threshold_m,
    )

    zed.start()

    logs = {
        "z_median": {"frames_total": 0, "pointing_frames": 0, "target_points": [], "infer_ms": [], "elapsed_sec": 0.0},
        "ransac": {"frames_total": 0, "pointing_frames": 0, "target_points": [], "infer_ms": [], "elapsed_sec": 0.0},
    }

    start = time.time()
    end = start + args.seconds
    try:
        while time.time() < end:
            rgb, _, point_cloud = zed.get_frames()
            if rgb is None or point_cloud is None:
                continue

            for name, rec in (("z_median", rec_z), ("ransac", rec_r)):
                t0 = time.perf_counter()
                state = rec.recognize(rgb, point_cloud)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                logs[name]["frames_total"] += 1
                logs[name]["infer_ms"].append(infer_ms)

                if state.get("label") == "POINTING":
                    logs[name]["pointing_frames"] += 1
                target = state.get("pointing_target_3d")
                if target is not None:
                    logs[name]["target_points"].append(target)
    finally:
        elapsed = time.time() - start
        logs["z_median"]["elapsed_sec"] = elapsed
        logs["ransac"]["elapsed_sec"] = elapsed
        rec_z.close()
        rec_r.close()
        zed.stop()

    summary = {
        "config": {
            "seconds": args.seconds,
            "fps": args.fps,
            "ransac_iterations": args.ransac_iterations,
            "ransac_threshold_m": args.ransac_threshold_m,
        },
        "z_median": evaluate(logs["z_median"]),
        "ransac": evaluate(logs["ransac"]),
    }

    summary["delta_ransac_minus_z_median"] = {
        "target_detected_ratio": round(summary["ransac"]["target_detected_ratio"] - summary["z_median"]["target_detected_ratio"], 4),
        "target_jitter_m": (
            round(summary["ransac"]["target_jitter_m"] - summary["z_median"]["target_jitter_m"], 5)
            if summary["ransac"]["target_jitter_m"] is not None and summary["z_median"]["target_jitter_m"] is not None
            else None
        ),
        "mean_inference_ms": round(summary["ransac"]["mean_inference_ms"] - summary["z_median"]["mean_inference_ms"], 3),
    }

    out_path = out_root / "ablation_plane_fitting.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] Ablation report generated")
    print(f" - output: {out_path}")
    print(f" - z_median: {summary['z_median']}")
    print(f" - ransac:   {summary['ransac']}")
    print(f" - delta:    {summary['delta_ransac_minus_z_median']}")


if __name__ == "__main__":
    main()
