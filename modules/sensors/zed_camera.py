import sys
import pyzed.sl as sl
import cv2

class ZEDCameraManager:
    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30, depth_mode=sl.DEPTH_MODE.NEURAL):
        """
        ZED 카메라 초기화 및 파라미터 설정
        VLA 모델(SmolVLA)과 모호성 해결을 위한 빠른 응답성을 위해 FPS와 해상도를 타협점으로 설정합니다.
        """
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        
        # 카메라 해상도 및 FPS 설정 (ZED X Mini 사양에 맞춤)
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        
        # Depth 모드 설정: PERFORMANCE는 deprecated 되었으므로 NEURAL을 기본 사용
        self.init_params.depth_mode = depth_mode
        self.init_params.coordinate_units = sl.UNIT.METER # 거리를 미터 단위로 통일
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        
        # 데이터 수신을 위한 런타임 파라미터 및 Mat 객체 준비
        self.runtime_params = sl.RuntimeParameters()
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.point_cloud_mat = sl.Mat()

    def start(self):
        """카메라 구동 시작"""
        status = self.camera.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[Error] ZED 카메라를 열 수 없습니다: {repr(status)}")
            sys.exit(1)
        print("[Info] ZED 카메라가 성공적으로 연결되었습니다.")

    def get_frames(self):
        """
        최신 프레임을 캡처하여 RGB, Depth, Point Cloud를 Numpy 배열로 반환
        Returns:
            rgb_image (np.ndarray): (H, W, 4) BGRA 이미지 (MediaPipe 및 모델 입력용)
            depth_map (np.ndarray): (H, W) 각 픽셀의 깊이 값 (m)
            point_cloud (np.ndarray): (H, W, 4) X, Y, Z, Color 값
        """
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 1. 왼쪽 렌즈의 RGB 이미지 (LeRobot 및 MediaPipe 용)
            self.camera.retrieve_image(self.image_mat, sl.VIEW.LEFT)
            rgb_image = self.image_mat.get_data()
            
            # 2. Depth 맵 (거리 측정용)
            self.camera.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
            depth_map = self.depth_mat.get_data()
            
            # 3. 3D Point Cloud (MediaPipe 2D 좌표를 3D 공간으로 변환할 때 핵심!)
            self.camera.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZRGBA)
            point_cloud = self.point_cloud_mat.get_data()
            
            return rgb_image, depth_map, point_cloud
        else:
            return None, None, None

    def get_camera_intrinsics(self):
        """카메라 내부 파라미터 반환 (2D -> 3D 투영 시 필요할 수 있음)"""
        calibration_params = self.camera.get_camera_information().camera_configuration.calibration_parameters
        left_cam = calibration_params.left_cam
        return {
            "fx": left_cam.fx,
            "fy": left_cam.fy,
            "cx": left_cam.cx,
            "cy": left_cam.cy
        }

    def stop(self):
        """카메라 자원 해제"""
        self.camera.close()
        print("[Info] ZED 카메라 연결이 해제되었습니다.")

# 단독 실행 및 테스트용 코드
if __name__ == "__main__":
    zed = ZEDCameraManager()
    zed.start()
    
    try:
        while True:
            rgb, depth, pc = zed.get_frames()
            
            if rgb is not None:
                # OpenCV는 BGR을 사용하므로 BGRA에서 BGR로 변환하여 출력
                display_img = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
                
                # 화면 중앙의 Depth 값 출력 (테스트용)
                h, w = depth.shape
                center_depth = depth[h//2, w//2]
                
                cv2.putText(display_img, f"Center Depth: {center_depth:.2f} m", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("ZED X Mini - RGB Stream", display_img)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        zed.stop()