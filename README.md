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

## Baseline (외부 레포)

`baseline/`은 용량 문제로 이 레포에 포함하지 않습니다. 아래 커맨드로 동일한 커밋을 받아오세요.

```bash
git clone https://github.com/amap-cvlab/ABot-Manipulation.git baseline/ABot-Manipulation
git -C baseline/ABot-Manipulation checkout bd32a886f6a61495d3a63a1861e7b8ea94310266

git clone https://github.com/facebookresearch/vggt.git baseline/vggt
git -C baseline/vggt checkout 44b3afbd1869d8bde4894dd8ea1e293112dd5eba
```
