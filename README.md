# SOArm VLA Project

ZED 카메라 + MediaPipe 제스처 인식 + Qwen2-VL + SmolVLA를 결합한 SOArm-101 로봇 팔 제어 파이프라인.

프로젝트 구조는 [directory_overview.md](directory_overview.md) 참고.

## Data

수집된 에피소드/캘리브레이션 데이터는 용량 문제로 이 레포에 포함하지 않고 HuggingFace Datasets에 업로드되어 있습니다.

- Dataset: https://huggingface.co/datasets/kyne0127/smol-vla-proj-data (private)

로컬에 받아오려면:

```bash
hf download kyne0127/smol-vla-proj-data --repo-type dataset --local-dir data
```
