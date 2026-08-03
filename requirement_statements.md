# 📄 프로젝트 명세서: SOArm-101 멀티모달 제스처-자연어 로봇 제어 시스템

## 1. 프로젝트 개요 (Overview)

* **프로젝트명:** SOArm-101 기반 모호성 해결형 VLA 제어 시스템
* **목표:** 자연어 명령어의 태생적 모호성("저거 집어", "파란 블록 집어")을 사용자의 Deictic 제스처(Pointing)와 Semantic 제스처(Thumbs-up, Fist)를 통해 해소하고, 이를 End-to-End 로봇 제어 정책(SmolVLA)과 결합하여 정확한 태스크 수행 능력을 확보한다.
* **핵심 가치:** * **Zero-Ambiguity:** VLM을 활용한 실시간 사용자 의도 파악 및 프롬프트 구체화.
* **End-to-End Control:** 비전-언어-행동(VLA) 모델을 통한 직접적인 로봇 관절(Joint) 제어.
* **Low Latency:** 300ms 이하의 응답 속도를 가진 멀티모달 파이프라인 구축.



## 2. 시스템 요구사항 (System Requirements)

### 2.1. 하드웨어 스펙

* **로봇 매니퓰레이터:** SOArm-101 (CAN bus 기반, `/dev/ttyACM0` 시리얼 통신)
* **메인 비전 센서:** ZED X Mini (Top-down 설치, 거리 1.2m)
* 해상도/프레임: 640x360 @ 30fps (HD720 기반 크롭/리사이즈)
* Depth 범위: 0.5m ~ 3.0m


* **컴퓨팅 노드 (추론 및 제어):** NVIDIA Jetson Orin 시리즈 또는 고성능 데스크탑 (최소 RTX 3090 / A100 파인튜닝용)

### 2.2. 소프트웨어 스펙

* **OS:** Ubuntu 22.04 LTS
* **Language:** Python 3.12+
* **Core Libraries:** * `lerobot` (Hugging Face) - 데이터 수집 및 VLA 정책 학습/추론
* `mediapipe` (v0.10+) - 2D 핸드 랜드마크 및 제스처 인식
* `pyzed` (ZED SDK 4.1+) - 3D Point Cloud 및 Depth 추출
* `transformers`, `qwen-vl-utils` - Qwen2-VL 프롬프트 정제 모델
* `PyTorch` - 딥러닝 백엔드



## 3. 핵심 모듈 명세 (Module Specifications)

### Module 1: 비전 센서 인터페이스 (`zed_camera.py`)

* **역할:** 물리적 세계의 시각 및 공간 데이터를 시스템에 공급.
* **Input:** Hardware Camera Stream
* **Output (30Hz):** * `rgb_image`: (H, W, 4) 해상도의 BGRA 배열
* `depth_map`: (H, W) 픽셀별 M(미터) 단위 거리 데이터
* `point_cloud`: (H, W, 4) 카메라 좌표계 기준 3D 공간 데이터 (X, Y, Z, C)



### Module 2: 공간 제스처 인식기 (`gesture_recognizer.py`)

* **역할:** 이미지에서 사용자의 손을 탐지하고, 가리키는 3D 방향 벡터와 의미론적 제스처를 추출.
* **Input:** `rgb_image`, `point_cloud` (from Module 1)
* **Output (이벤트 발생 시):**
* `gesture_state`: Dictionary 형태
* `label`: "POINTING", "THUMBS_UP", "FIST", "NONE"
* `ray_vector`: `[x, y, z]` (로봇 Base 좌표계 기준 가리키는 방향)
* `confidence`: 0.0 ~ 1.0 (임계값 0.85 이상 시 유효 처리)


### Module 3: VLM 프롬프트 정제기 (`query_refiner.py`)

* **역할:** 모호한 인간의 언어와 물리적 제스처(3D 벡터)를 결합하여 명확한 텍스트 지시어로 변환.
* **Input:** * 사용자 초기 발화 (예: "저 파란 컵 치워줘")
* `gesture_state` (예: `{"label": "POINTING", "ray_vector": [0.4, 0.1, 0.0]}`)


* **Output:** Refined Prompt (예: "Pick up the blue cup located at spatial coordinate [0.4, 0.1, 0.0]")

### Module 4: VLA 액션 추론기 (`policy_runner.py`)

* **역할:** 정제된 텍스트와 현재 시각 정보를 바탕으로 로봇의 다음 목표 관절 각도를 계산. (SmolVLA 기반)
* **Input (10Hz):** Refined Prompt (from Module 3) + `rgb_image` (from Module 1)
* **Output:** `action_tensor`: `[q1, q2, q3, q4, q5, q6, gripper]` (목표 조인트 라디안 및 그리퍼 상태)

### Module 5: 하드웨어 컨트롤러 (`soarm_driver.py`)

* **역할:** VLA 모델이 예측한 액션을 실제 모터 제어 신호로 변환. 긴급 정지(Fist) 제어 통합.
* **Input:** `action_tensor` (from Module 4) OR 하드웨어 인터럽트 신호
* **Output:** SOArm-101 직렬 통신 패킷 전송.

## 4. 개발 로드맵 (Phases)

| 단계 | 목표 | 세부 마일스톤 | 예상 소요 |
| --- | --- | --- | --- |
| **Phase 1** | **베이스라인 구축** | 1. ZED Camera / SOArm-101 하드웨어 연동 스크립트 작성<br>

<br>2. LeRobot 기반 기본 데이터(언어+조작) 수집 파이프라인 구축<br>

<br>3. 순수 언어 기반 Pick-and-Place 테스트 | 1주 |
| **Phase 2** | **제스처 모듈 개발** | 1. MediaPipe + Point Cloud 융합 3D Ray 추정 로직 구현<br>

<br>2. 카메라-로봇 간 좌표계 변환(Calibration) 수행<br>

<br>3. Semantic 제스처(Thumbs-up/Fist) 인식 및 인터럽트 연동 | 2주 |
| **Phase 3** | **End-to-End 통합** | 1. Qwen2-VL을 이용한 Refiner 프롬프트 엔지니어링<br>

<br>2. 제스처 조건이 포함된 데이터셋(5k) 구축<br>

<br>3. SmolVLA LoRA 파인튜닝 (A100 활용) | 2주 |
| **Phase 4** | **최적화 및 평가** | 1. 모듈 간 비동기 파이프라인(core_loop) 통합<br>

<br>2. End-to-End Latency < 300ms 최적화<br>

<br>3. 모호성 해결 성공률 등 정량 지표 측정 | 1주 |

## 5. 핵심 평가 지표 (KPIs)

| 평가 항목 | 측정 방법 
| --- | --- |
| **Task Success Rate** | 단일 객체 Pick-and-Place 성공 횟수 / 시도 횟수 |
| **Target Localization** | Pointing 시 추정된 3D 좌표와 실제 객체 좌표 간 오차 |
| **Ambiguity Resolution** | 모호한 지시어("저것", "왼쪽") 입력 시 정확한 타겟팅 성공률 |
| **System Latency** | 카메라 프레임 입력부터 첫 관절 제어 신호 발생까지의 시간 |