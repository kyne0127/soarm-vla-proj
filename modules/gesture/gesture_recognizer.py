from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class GestureState:
	label: str
	ray_vector: list[float] | None
	pointing_3d: list[float] | None
	pointing_target_3d: list[float] | None
	ray_vector_robot: list[float] | None
	pointing_3d_robot: list[float] | None
	pointing_target_3d_robot: list[float] | None
	confidence: float
	handedness: str
	fingertip_px: list[int] | None

	def as_dict(self) -> dict[str, Any]:
		return {
			"label": self.label,
			"ray_vector": self.ray_vector,
			"pointing_3d": self.pointing_3d,
			"pointing_target_3d": self.pointing_target_3d,
			"ray_vector_robot": self.ray_vector_robot,
			"pointing_3d_robot": self.pointing_3d_robot,
			"pointing_target_3d_robot": self.pointing_target_3d_robot,
			"confidence": self.confidence,
			"handedness": self.handedness,
			"fingertip_px": self.fingertip_px,
		}


class GestureRecognizer:
	"""Estimate semantic hand gesture and 3D pointing target from RGB + point cloud."""

	def __init__(
		self,
		hand_model_path: str = "models/hand_landmarker.task",
		camera_to_robot_path: str = "data/calibration/camera_to_robot.json",
		confidence_threshold: float = 0.85,
		plane_method: str = "z_median",
		ransac_iterations: int = 120,
		ransac_threshold_m: float = 0.02,
	):
		self.confidence_threshold = confidence_threshold
		if plane_method not in {"z_median", "ransac"}:
			raise ValueError("plane_method must be 'z_median' or 'ransac'")
		self.plane_method = plane_method
		self.ransac_iterations = ransac_iterations
		self.ransac_threshold_m = ransac_threshold_m
		self._rng = np.random.default_rng()
		self._detector = self._build_detector(Path(hand_model_path))
		self._connections = self._get_hand_connections()
		self._t_camera_to_robot = self._load_camera_to_robot(Path(camera_to_robot_path))

	@staticmethod
	def _load_camera_to_robot(path: Path) -> np.ndarray | None:
		if not path.exists():
			return None

		if path.suffix.lower() == ".npy":
			mat = np.load(path)
		else:
			with path.open("r", encoding="utf-8") as f:
				payload = json.load(f)
			if isinstance(payload, dict):
				if "T_camera_to_robot" in payload:
					payload = payload["T_camera_to_robot"]
				elif "camera_to_robot" in payload:
					payload = payload["camera_to_robot"]
			mat = np.asarray(payload, dtype=np.float64)

		if mat.shape != (4, 4):
			raise ValueError(f"camera_to_robot must be 4x4. got {mat.shape} from {path}")
		return mat.astype(np.float64)

	def _to_robot_point(self, point_xyz: np.ndarray) -> np.ndarray | None:
		if self._t_camera_to_robot is None:
			return None
		h = np.ones(4, dtype=np.float64)
		h[:3] = point_xyz.astype(np.float64)
		out = self._t_camera_to_robot @ h
		return out[:3]

	def _to_robot_vector(self, ray_xyz: np.ndarray) -> np.ndarray | None:
		if self._t_camera_to_robot is None:
			return None
		rot = self._t_camera_to_robot[:3, :3]
		vec = rot @ ray_xyz.astype(np.float64)
		return self._normalize(vec)

	def _build_detector(self, hand_model_path: Path):
		if hasattr(mp, "solutions"):
			return mp.solutions.hands.Hands(
				static_image_mode=False,
				max_num_hands=2,
				model_complexity=1,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5,
			)

		if not hand_model_path.exists():
			raise FileNotFoundError(
				"MediaPipe tasks backend requires hand_landmarker.task. "
				f"Missing file: {hand_model_path}"
			)

		from mediapipe.tasks import python as mp_tasks
		from mediapipe.tasks.python import vision

		options = vision.HandLandmarkerOptions(
			base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model_path)),
			num_hands=2,
			min_hand_detection_confidence=0.5,
			min_hand_presence_confidence=0.5,
			min_tracking_confidence=0.5,
		)
		return vision.HandLandmarker.create_from_options(options)

	def _get_hand_connections(self):
		if hasattr(mp, "solutions"):
			return mp.solutions.hands.HAND_CONNECTIONS

		from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

		return HandLandmarksConnections.HAND_CONNECTIONS

	@staticmethod
	def _is_valid_xyz(xyz: np.ndarray) -> bool:
		return bool(np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e-6)

	@staticmethod
	def _sample_point_cloud(point_cloud: np.ndarray, x: int, y: int, radius: int = 3) -> np.ndarray | None:
		h, w, _ = point_cloud.shape
		x0 = max(0, x - radius)
		x1 = min(w - 1, x + radius)
		y0 = max(0, y - radius)
		y1 = min(h - 1, y + radius)

		patch = point_cloud[y0 : y1 + 1, x0 : x1 + 1, :3].reshape(-1, 3)
		valid = patch[np.isfinite(patch).all(axis=1)]
		if len(valid) == 0:
			return None
		return np.median(valid, axis=0)

	@staticmethod
	def _to_pixel(x_norm: float, y_norm: float, w: int, h: int) -> tuple[int, int]:
		x = int(np.clip(x_norm * w, 0, w - 1))
		y = int(np.clip(y_norm * h, 0, h - 1))
		return x, y

	@staticmethod
	def _normalize(vec: np.ndarray) -> np.ndarray | None:
		n = float(np.linalg.norm(vec))
		if n < 1e-8:
			return None
		return vec / n

	@staticmethod
	def _plane_points_from_roi(point_cloud: np.ndarray) -> np.ndarray:
		h, w, _ = point_cloud.shape
		x0 = int(w * 0.25)
		x1 = int(w * 0.75)
		y0 = int(h * 0.60)
		y1 = int(h * 0.95)
		roi = point_cloud[y0:y1, x0:x1, :3].reshape(-1, 3)
		valid = roi[np.isfinite(roi).all(axis=1)]
		return valid

	@staticmethod
	def _fit_plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float] | None:
		if len(points) < 3:
			return None
		centroid = np.mean(points, axis=0)
		centered = points - centroid
		_, _, vh = np.linalg.svd(centered, full_matrices=False)
		normal = vh[-1]
		normal = normal / np.linalg.norm(normal)
		d = -float(np.dot(normal, centroid))
		return normal.astype(np.float64), d

	def _estimate_table_plane_z_median(self, point_cloud: np.ndarray) -> tuple[np.ndarray, float] | None:
		"""Approximate tabletop as z=constant using median z in ROI."""
		points = self._plane_points_from_roi(point_cloud)
		if len(points) < 50:
			return None
		z_plane = float(np.median(points[:, 2]))
		normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
		d = -z_plane
		return normal, d

	def _estimate_table_plane_ransac(self, point_cloud: np.ndarray) -> tuple[np.ndarray, float] | None:
		"""Fit tabletop plane ax+by+cz+d=0 with RANSAC over ROI points."""
		points = self._plane_points_from_roi(point_cloud)
		if len(points) < 150:
			return None

		best_inliers = None
		best_count = 0
		n_points = len(points)

		for _ in range(self.ransac_iterations):
			idx = self._rng.choice(n_points, size=3, replace=False)
			p1, p2, p3 = points[idx]
			n = np.cross(p2 - p1, p3 - p1)
			norm = np.linalg.norm(n)
			if norm < 1e-8:
				continue
			n = n / norm
			d = -float(np.dot(n, p1))
			dists = np.abs(points @ n + d)
			inliers = dists < self.ransac_threshold_m
			count = int(np.sum(inliers))
			if count > best_count:
				best_count = count
				best_inliers = inliers

		if best_inliers is None or best_count < 80:
			return None

		refined = self._fit_plane_from_points(points[best_inliers])
		if refined is None:
			return None
		normal, d = refined

		# Keep normal orientation stable: prefer +z normal.
		if normal[2] < 0:
			normal = -normal
			d = -d
		return normal, d

	def _estimate_table_plane(self, point_cloud: np.ndarray) -> tuple[np.ndarray, float] | None:
		if self.plane_method == "ransac":
			plane = self._estimate_table_plane_ransac(point_cloud)
			if plane is not None:
				return plane
			# Fallback for robustness when RANSAC fails.
		return self._estimate_table_plane_z_median(point_cloud)

	@staticmethod
	def _intersect_ray_with_plane(origin: np.ndarray, ray: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray | None:
		"""Compute intersection between ray p(t)=origin+t*ray and plane n·p+d=0."""
		den = float(np.dot(normal, ray))
		if abs(den) < 1e-8:
			return None
		t = -float(np.dot(normal, origin) + d) / den
		if t <= 0:
			return None
		return origin + t * ray

	def _detect_with_solutions(self, bgr: np.ndarray):
		rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		result = self._detector.process(rgb)

		hands = []
		if not result.multi_hand_landmarks:
			return hands

		for idx, lmset in enumerate(result.multi_hand_landmarks):
			handedness = "Unknown"
			score = 0.0
			if result.multi_handedness and idx < len(result.multi_handedness):
				cls = result.multi_handedness[idx].classification
				if cls:
					handedness = cls[0].label
					score = float(cls[0].score)

			landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lmset.landmark], dtype=np.float32)
			hands.append({"landmarks": landmarks, "handedness": handedness, "score": score})
		return hands

	def _detect_with_tasks(self, bgr: np.ndarray):
		rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		result = self._detector.detect(mp_image)

		hands = []
		if not result.hand_landmarks:
			return hands

		for idx, lmset in enumerate(result.hand_landmarks):
			handedness = "Unknown"
			score = 0.0
			if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
				handedness = result.handedness[idx][0].category_name
				score = float(result.handedness[idx][0].score)

			landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lmset], dtype=np.float32)
			hands.append({"landmarks": landmarks, "handedness": handedness, "score": score})
		return hands

	def _detect_hands(self, bgr: np.ndarray):
		if hasattr(mp, "solutions") and hasattr(self._detector, "process"):
			return self._detect_with_solutions(bgr)
		return self._detect_with_tasks(bgr)

	def _estimate_pointing_for_hand(self, hand: dict[str, Any], point_cloud: np.ndarray) -> GestureState:
		h, w, _ = point_cloud.shape
		landmarks = hand["landmarks"]

		# Index fingertip(8) and MCP(5) are used for deictic pointing direction.
		tip_x, tip_y = self._to_pixel(landmarks[8, 0], landmarks[8, 1], w, h)
		mcp_x, mcp_y = self._to_pixel(landmarks[5, 0], landmarks[5, 1], w, h)

		tip_xyz = self._sample_point_cloud(point_cloud, tip_x, tip_y, radius=3)
		mcp_xyz = self._sample_point_cloud(point_cloud, mcp_x, mcp_y, radius=3)

		if tip_xyz is None or mcp_xyz is None:
			return GestureState(
				label="NONE",
				ray_vector=None,
				pointing_3d=None,
				pointing_target_3d=None,
				ray_vector_robot=None,
				pointing_3d_robot=None,
				pointing_target_3d_robot=None,
				confidence=0.0,
				handedness=hand["handedness"],
				fingertip_px=[tip_x, tip_y],
			)

		ray = self._normalize(tip_xyz - mcp_xyz)
		if ray is None:
			tip_robot = self._to_robot_point(tip_xyz)
			return GestureState(
				label="NONE",
				ray_vector=None,
				pointing_3d=tip_xyz.tolist(),
				pointing_target_3d=None,
				ray_vector_robot=None,
				pointing_3d_robot=tip_robot.tolist() if tip_robot is not None else None,
				pointing_target_3d_robot=None,
				confidence=0.0,
				handedness=hand["handedness"],
				fingertip_px=[tip_x, tip_y],
			)

		# Use hand classification confidence and geometric consistency jointly.
		finger_2d = np.array([tip_x - mcp_x, tip_y - mcp_y], dtype=np.float32)
		finger_2d_norm = float(np.linalg.norm(finger_2d))
		geom_conf = min(1.0, finger_2d_norm / 60.0)
		conf = float(0.6 * hand["score"] + 0.4 * geom_conf)

		label = "POINTING" if conf >= self.confidence_threshold else "NONE"
		plane = self._estimate_table_plane(point_cloud)
		target_xyz = (
			self._intersect_ray_with_plane(tip_xyz, ray, plane[0], plane[1]) if plane is not None else None
		)
		tip_robot = self._to_robot_point(tip_xyz)
		ray_robot = self._to_robot_vector(ray)
		target_robot = self._to_robot_point(target_xyz) if target_xyz is not None else None
		return GestureState(
			label=label,
			ray_vector=ray.astype(float).tolist(),
			pointing_3d=tip_xyz.astype(float).tolist(),
			pointing_target_3d=target_xyz.astype(float).tolist() if target_xyz is not None else None,
			ray_vector_robot=ray_robot.astype(float).tolist() if ray_robot is not None else None,
			pointing_3d_robot=tip_robot.astype(float).tolist() if tip_robot is not None else None,
			pointing_target_3d_robot=target_robot.astype(float).tolist() if target_robot is not None else None,
			confidence=round(conf, 4),
			handedness=hand["handedness"],
			fingertip_px=[tip_x, tip_y],
		)

	def recognize(self, rgb_bgra: np.ndarray, point_cloud: np.ndarray) -> dict[str, Any]:
		"""Return best gesture state from current frame.

		Args:
			rgb_bgra: (H, W, 4) BGRA frame from ZED.
			point_cloud: (H, W, 4) XYZRGBA from ZED.
		"""
		if rgb_bgra is None or point_cloud is None:
			return GestureState("NONE", None, None, None, None, None, None, 0.0, "Unknown", None).as_dict()

		bgr = cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR)
		hands = self._detect_hands(bgr)
		if not hands:
			return GestureState("NONE", None, None, None, None, None, None, 0.0, "Unknown", None).as_dict()

		states = [self._estimate_pointing_for_hand(hand, point_cloud) for hand in hands]
		best = max(states, key=lambda s: s.confidence)
		return best.as_dict()

	def close(self):
		if hasattr(self._detector, "close"):
			self._detector.close()


if __name__ == "__main__":
	import sys

	from modules.sensors.zed_camera import ZEDCameraManager

	zed = ZEDCameraManager()
	rec = GestureRecognizer()
	zed.start()

	try:
		while True:
			rgb, _, pc = zed.get_frames()
			state = rec.recognize(rgb, pc)
			if rgb is None:
				continue

			frame = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
			text = (
				f"{state['label']} conf={state['confidence']:.2f} "
				f"hand={state['handedness']}"
			)
			cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

			if state["pointing_3d_robot"] is not None:
				rx, ry, rz = state["pointing_3d_robot"]
				robot_text = f"robot xyz=({rx:.3f}, {ry:.3f}, {rz:.3f})"
				cv2.putText(
					frame,
					robot_text,
					(20, 72),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.65,
					(255, 255, 255),
					2,
				)

			if state["fingertip_px"] is not None:
				fx, fy = state["fingertip_px"]
				cv2.circle(frame, (fx, fy), 7, (0, 0, 255), 2)

			cv2.imshow("GestureRecognizer Debug", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	except KeyboardInterrupt:
		pass
	finally:
		rec.close()
		zed.stop()
		cv2.destroyAllWindows()
		sys.exit(0)
