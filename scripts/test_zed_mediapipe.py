import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from modules.sensors.zed_camera import ZEDCameraManager
from modules.gesture.gesture_recognizer import GestureRecognizer


class HandLandmarkOverlay:
	def __init__(self, hand_model_path: Path):
		self.backend = None
		self._hands = None
		self._detector = None
		self._connections = None

		if hasattr(mp, "solutions"):
			self._mp_hands = mp.solutions.hands
			self._mp_draw = mp.solutions.drawing_utils
			self._mp_style = mp.solutions.drawing_styles
			self._hands = self._mp_hands.Hands(
				static_image_mode=False,
				max_num_hands=2,
				model_complexity=1,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5,
			)
			self.backend = "solutions"
			return

		if not hand_model_path.exists():
			raise FileNotFoundError(
				"MediaPipe tasks backend requires a hand landmarker model file. "
				f"Missing: {hand_model_path}"
			)

		from mediapipe.tasks import python as mp_tasks
		from mediapipe.tasks.python import vision
		from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

		options = vision.HandLandmarkerOptions(
			base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model_path)),
			num_hands=2,
			min_hand_detection_confidence=0.5,
			min_hand_presence_confidence=0.5,
			min_tracking_confidence=0.5,
		)
		self._detector = vision.HandLandmarker.create_from_options(options)
		self._connections = HandLandmarksConnections.HAND_CONNECTIONS
		self.backend = "tasks"

	def draw(self, bgr_frame):
		annotated = bgr_frame.copy()
		h, w = annotated.shape[:2]
		hand_infos = []

		if self.backend == "solutions":
			rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
			result = self._hands.process(rgb_frame)

			hand_count = 0
			if result.multi_hand_landmarks:
				hand_count = len(result.multi_hand_landmarks)
				for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
					self._mp_draw.draw_landmarks(
						annotated,
						hand_landmarks,
						self._mp_hands.HAND_CONNECTIONS,
						self._mp_style.get_default_hand_landmarks_style(),
						self._mp_style.get_default_hand_connections_style(),
					)

					handedness = "Unknown"
					if result.multi_handedness and idx < len(result.multi_handedness):
						cls = result.multi_handedness[idx].classification
						if cls:
							handedness = cls[0].label

					wrist = hand_landmarks.landmark[0]
					x = int(np.clip(wrist.x * w, 0, w - 1))
					y = int(np.clip(wrist.y * h, 0, h - 1))
					hand_infos.append({"handedness": handedness, "wrist_xy": (x, y)})
			return annotated, hand_count, hand_infos

		if self.backend == "tasks":
			rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
			result = self._detector.detect(mp_image)

			hand_count = 0
			if result.hand_landmarks:
				hand_count = len(result.hand_landmarks)
				for idx, hand_landmarks in enumerate(result.hand_landmarks):
					points = []
					for lm in hand_landmarks:
						x = int(np.clip(lm.x * w, 0, w - 1))
						y = int(np.clip(lm.y * h, 0, h - 1))
						points.append((x, y))
						cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

					for conn in self._connections:
						start = conn.start
						end = conn.end
						if 0 <= start < len(points) and 0 <= end < len(points):
							cv2.line(annotated, points[start], points[end], (255, 180, 0), 2)

					handedness = "Unknown"
					if result.handedness and idx < len(result.handedness):
						cats = result.handedness[idx]
						if cats:
							handedness = cats[0].category_name

					if points:
						hand_infos.append({"handedness": handedness, "wrist_xy": points[0]})
			return annotated, hand_count, hand_infos

		return annotated, 0, hand_infos

	def close(self):
		if self._hands is not None:
			self._hands.close()
		if self._detector is not None:
			self._detector.close()


def parse_args():
	parser = argparse.ArgumentParser(
		description="Save ZED test video/images under data/test_runs for quick quality checks."
	)
	parser.add_argument("--seconds", type=int, default=10, help="Capture duration in seconds")
	parser.add_argument("--fps", type=int, default=30, help="Target camera FPS")
	parser.add_argument(
		"--snapshot-interval",
		type=float,
		default=1.0,
		help="Seconds between snapshot saves",
	)
	parser.add_argument(
		"--min-valid-ratio",
		type=float,
		default=0.95,
		help="Minimum valid frame ratio for PASS",
	)
	parser.add_argument(
		"--min-fps-ratio",
		type=float,
		default=0.85,
		help="Minimum effective_fps/requested_fps ratio for PASS",
	)
	parser.add_argument(
		"--hand-model-path",
		type=str,
		default="models/hand_landmarker.task",
		help="Path to MediaPipe Hand Landmarker .task model (used for tasks backend)",
	)
	parser.add_argument(
		"--plane-method",
		type=str,
		default="z_median",
		choices=["z_median", "ransac"],
		help="Table plane fitting method for pointing target intersection",
	)
	parser.add_argument(
		"--ransac-iterations",
		type=int,
		default=120,
		help="RANSAC iteration count (used when --plane-method=ransac)",
	)
	parser.add_argument(
		"--ransac-threshold-m",
		type=float,
		default=0.02,
		help="RANSAC inlier distance threshold in meters",
	)
	return parser.parse_args()


def build_run_dirs(project_root: Path) -> tuple[Path, Path, Path]:
	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_root = project_root / "data" / "test_runs" / run_id
	image_dir = run_root / "images"
	depth_dir = run_root / "depth"

	image_dir.mkdir(parents=True, exist_ok=True)
	depth_dir.mkdir(parents=True, exist_ok=True)
	return run_root, image_dir, depth_dir


def normalize_depth_for_preview(depth):
	finite_mask = np.isfinite(depth)
	if not np.any(finite_mask):
		return np.zeros(depth.shape, dtype="uint8")

	safe_depth = np.where(finite_mask, depth, 0.0)
	clipped = cv2.normalize(safe_depth, None, 0, 255, cv2.NORM_MINMAX)
	return clipped.astype("uint8")


def project_xyz_to_pixel(point_xyz, intrinsics, width, height):
	if point_xyz is None:
		return None

	x, y, z = point_xyz
	if not np.isfinite([x, y, z]).all() or abs(z) < 1e-8:
		return None

	fx = intrinsics["fx"]
	fy = intrinsics["fy"]
	cx = intrinsics["cx"]
	cy = intrinsics["cy"]

	u = int(round(fx * (x / z) + cx))
	v = int(round(fy * (y / z) + cy))
	if not (0 <= u < width and 0 <= v < height):
		return None
	return u, v


def get_depth_at_pixel(depth, x, y):
	h, w = depth.shape
	px = int(np.clip(x, 0, w - 1))
	py = int(np.clip(y, 0, h - 1))
	value = float(depth[py, px])
	if not np.isfinite(value) or value <= 0:
		return None
	return value


def evaluate_quality(metrics, min_valid_ratio: float, min_fps_ratio: float):
	frames_total = metrics["frames_total"]
	frames_valid = metrics["frames_valid"]
	effective_fps = metrics["effective_fps"]
	requested_fps = metrics["requested_fps"]

	valid_ratio = (frames_valid / frames_total) if frames_total > 0 else 0.0
	fps_ratio = (effective_fps / requested_fps) if requested_fps > 0 else 0.0

	checks = {
		"valid_ratio_check": {
			"value": round(valid_ratio, 4),
			"threshold": min_valid_ratio,
			"pass": valid_ratio >= min_valid_ratio,
		},
		"fps_ratio_check": {
			"value": round(fps_ratio, 4),
			"threshold": min_fps_ratio,
			"pass": fps_ratio >= min_fps_ratio,
		},
	}

	passed_count = sum(1 for c in checks.values() if c["pass"])
	if passed_count == len(checks):
		status = "PASS"
	elif passed_count > 0:
		status = "WARN"
	else:
		status = "FAIL"

	hints = []
	if not checks["valid_ratio_check"]["pass"]:
		hints.append(
			"valid_ratio가 낮습니다. USB 대역폭/케이블 상태를 확인하고, 카메라 재연결 후 재측정하세요."
		)
		hints.append(
			"프레임 드롭이 지속되면 해상도/FPS를 낮춰(예: HD720@15) 안정성을 먼저 확보하세요."
		)

	if not checks["fps_ratio_check"]["pass"]:
		hints.append(
			"effective_fps가 목표 대비 낮습니다. 다른 GPU/CPU 작업을 종료하고, 캡처 프로세스 우선순위를 확보하세요."
		)
		hints.append(
			"NEURAL depth 품질이 과하면 FPS가 떨어질 수 있습니다. 목표 지연에 맞춰 FPS 또는 후처리를 조정하세요."
		)

	if metrics["center_depth_mean_m"] is None:
		hints.append(
			"유효 depth 샘플이 없습니다. 장면 조명/텍스처를 개선하고 렌즈 가림 여부를 확인하세요."
		)

	if status == "PASS":
		hints.append("성능이 기준을 만족합니다. 동일 환경에서 gesture 모듈 통합 테스트를 진행하세요.")

	return {
		"status": status,
		"checks": checks,
		"hints": hints,
	}


def main():
	args = parse_args()
	project_root = PROJECT_ROOT
	run_root, image_dir, depth_dir = build_run_dirs(project_root)
	hand_model_path = (project_root / args.hand_model_path).resolve()

	zed = ZEDCameraManager(fps=args.fps)
	zed.start()
	overlay = HandLandmarkOverlay(hand_model_path=hand_model_path)
	gesture = GestureRecognizer(
		hand_model_path=str(hand_model_path),
		plane_method=args.plane_method,
		ransac_iterations=args.ransac_iterations,
		ransac_threshold_m=args.ransac_threshold_m,
	)
	intrinsics = zed.get_camera_intrinsics()

	raw_video_path = run_root / "rgb_raw.mp4"
	overlay_video_path = run_root / "rgb_overlay.mp4"
	raw_writer = None
	overlay_writer = None

	start_ts = time.time()
	end_ts = start_ts + args.seconds
	last_snapshot_ts = 0.0

	frame_count = 0
	valid_count = 0
	depth_samples = []
	wrist_depth_samples = []
	target_detected_frames = 0
	detected_hand_frames = 0
	max_hands_seen = 0

	try:
		while time.time() < end_ts:
			frame_count += 1
			rgb, depth, point_cloud = zed.get_frames()
			if rgb is None or depth is None or point_cloud is None:
				continue

			valid_count += 1
			bgr = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
			annotated, hand_count, hand_infos = overlay.draw(bgr)
			gesture_state = gesture.recognize(rgb, point_cloud)
			target_px = project_xyz_to_pixel(
				gesture_state.get("pointing_target_3d"),
				intrinsics,
				annotated.shape[1],
				annotated.shape[0],
			)
			if target_px is not None:
				target_detected_frames += 1
				tx, ty = target_px
				cv2.drawMarker(
					annotated,
					(tx, ty),
					(0, 0, 255),
					markerType=cv2.MARKER_CROSS,
					markerSize=20,
					thickness=2,
				)
				cv2.putText(
					annotated,
					"TARGET",
					(tx + 10, max(20, ty - 10)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.65,
					(0, 0, 255),
					2,
				)
			if hand_count > 0:
				detected_hand_frames += 1
			max_hands_seen = max(max_hands_seen, hand_count)

			for hand_info in hand_infos:
				x, y = hand_info["wrist_xy"]
				handedness = hand_info["handedness"]
				wrist_depth = get_depth_at_pixel(depth, x, y)
				depth_text = f"{wrist_depth:.3f}m" if wrist_depth is not None else "N/A"
				if wrist_depth is not None:
					wrist_depth_samples.append(wrist_depth)

				cv2.circle(annotated, (x, y), 6, (0, 0, 255), 2)
				cv2.putText(
					annotated,
					f"{handedness}",
					(x + 10, max(20, y - 8)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(0, 255, 255),
					2,
				)
				cv2.putText(
					annotated,
					f"wrist z: {depth_text}",
					(x + 10, min(annotated.shape[0] - 10, y + 18)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(255, 255, 255),
					2,
				)

			cv2.putText(
				annotated,
				f"Hands: {hand_count}",
				(20, 36),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				(0, 255, 255),
				2,
			)

			if raw_writer is None or overlay_writer is None:
				h, w = bgr.shape[:2]
				raw_writer = cv2.VideoWriter(
					str(raw_video_path),
					cv2.VideoWriter_fourcc(*"mp4v"),
					args.fps,
					(w, h),
				)
				overlay_writer = cv2.VideoWriter(
					str(overlay_video_path),
					cv2.VideoWriter_fourcc(*"mp4v"),
					args.fps,
					(w, h),
				)

			raw_writer.write(bgr)
			overlay_writer.write(annotated)

			h, w = depth.shape
			center_depth = float(depth[h // 2, w // 2])
			depth_samples.append(center_depth)

			now = time.time()
			if now - last_snapshot_ts >= args.snapshot_interval:
				stamp = datetime.now().strftime("%H%M%S_%f")
				rgb_path = image_dir / f"rgb_{stamp}.jpg"
				depth_path = depth_dir / f"depth_{stamp}.png"

				cv2.imwrite(str(rgb_path), annotated)
				depth_preview = normalize_depth_for_preview(depth)
				cv2.imwrite(str(depth_path), depth_preview)
				last_snapshot_ts = now

	finally:
		if raw_writer is not None:
			raw_writer.release()
		if overlay_writer is not None:
			overlay_writer.release()
		gesture.close()
		overlay.close()
		zed.stop()

	elapsed = max(time.time() - start_ts, 1e-6)
	metrics = {
		"plane_method": args.plane_method,
		"ransac_iterations": args.ransac_iterations,
		"ransac_threshold_m": args.ransac_threshold_m,
		"duration_seconds": args.seconds,
		"elapsed_seconds": round(elapsed, 3),
		"requested_fps": args.fps,
		"frames_total": frame_count,
		"frames_valid": valid_count,
		"effective_fps": round(valid_count / elapsed, 3),
		"center_depth_mean_m": round(sum(depth_samples) / len(depth_samples), 4)
		if depth_samples
		else None,
		"wrist_depth_mean_m": round(sum(wrist_depth_samples) / len(wrist_depth_samples), 4)
		if wrist_depth_samples
		else None,
		"target_detected_frames": target_detected_frames,
		"target_detected_ratio": round((target_detected_frames / valid_count), 4)
		if valid_count > 0
		else 0.0,
		"hand_detected_frames": detected_hand_frames,
		"hand_detected_ratio": round((detected_hand_frames / valid_count), 4)
		if valid_count > 0
		else 0.0,
		"max_hands_seen": max_hands_seen,
		"overlay_backend": overlay.backend,
		"output_video_raw": str(raw_video_path),
		"output_video_overlay": str(overlay_video_path),
		"output_images_dir": str(image_dir),
		"output_depth_dir": str(depth_dir),
	}

	quality_report = evaluate_quality(
		metrics,
		min_valid_ratio=args.min_valid_ratio,
		min_fps_ratio=args.min_fps_ratio,
	)

	metrics_path = run_root / "metrics.json"
	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)

	quality_path = run_root / "quality_report.json"
	with quality_path.open("w", encoding="utf-8") as f:
		json.dump(quality_report, f, ensure_ascii=False, indent=2)

	print("[DONE] Test run saved")
	print(f" - run_root: {run_root}")
	print(f" - video(raw): {raw_video_path}")
	print(f" - video(overlay): {overlay_video_path}")
	print(f" - metrics: {metrics_path}")
	print(f" - quality: {quality_path}")
	print(f" - valid frames: {valid_count}/{frame_count}")
	print(f" - effective fps: {metrics['effective_fps']}")
	print(f" - status: {quality_report['status']}")
	for hint in quality_report["hints"]:
		print(f" - hint: {hint}")


if __name__ == "__main__":
	main()
