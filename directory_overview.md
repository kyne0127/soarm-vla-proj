soarm_vla_project/
│
├── configs/                     # 하드웨어 포트, 모델 하이퍼파라미터 등 환경 설정
│   ├── hardware_cfg.yaml        # 예: ZED 해상도, SOArm 통신 포트(/dev/ttyACM0)
│   └── model_cfg.yaml           # 예: Qwen2-VL 프롬프트, SmolVLA 추론 파라미터
│
├── data/                        # 수집된 에피소드 및 캘리브레이션 데이터
│   ├── calibration/             # 카메라-로봇 캘리브레이션 매트릭스 저장
│   └── episodes/                # LeRobot 포맷으로 저장된 데모 데이터
│
├── modules/                     # 시스템 아키텍처의 핵심 모듈 (독립적 실행 가능하도록 설계)
│   ├── __init__.py
│   ├── sensors/                 
│   │   └── zed_camera.py        # ZED X Mini 래퍼 (RGB, Depth, PointCloud 동기화)
│   ├── gesture/
│   │   └── gesture_recognizer.py# MediaPipe 기반 제스처 인식 및 3D Ray 계산
│   ├── vlm/
│   │   └── query_refiner.py     # Qwen2-VL 로드 및 이벤트 기반 프롬프트 생성 로직
│   ├── vla/
│   │   └── policy_runner.py     # SmolVLA 추론 및 액션 토큰 디코딩
│   └── robot/
│       └── soarm_driver.py      # SOArm-101 시리얼 통신 제어 및 상태 읽기
│
├── scripts/                     # 유틸리티 및 독립적인 실행 스크립트
│   ├── collect_data.py          # Phase 1: VLA 학습용 베이스라인 데이터 수집 
│   ├── test_zed_mediapipe.py    # Phase 2: 제스처 모듈 단독 테스트
│   └── train_lora.py            # Phase 3: SmolVLA 파인튜닝 스크립트
│
├── core_loop.py                 # 비동기 멀티프로세싱/스레딩으로 전체 모듈을 엮는 메인 파이프라인
├── requirements.txt             # 환경 재현을 위한 패키지 목록
└── README.md                    # 프로젝트 실행 방법 및 핀맵 문서화