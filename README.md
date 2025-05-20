# 핸드폰 파손 감지 AI 모델

이 프로젝트는 핸드폰 파손 정도를 감지하여 A, B, C, F 등급으로 분류하는 인공지능 모델을 구현합니다.

## 프로젝트 개요

- **목적**: 핸드폰 이미지를 분석하여 파손 정도를 자동으로 분류
- **분류 등급**:
  - A급: 완전 새거처럼 멀정함
  - B급: 약간의 생활 기스가 있음
  - C급: 디스플레이가 눈에 띄게 파손됨
  - F급: 사용불가능할 정도로 파손됨
- **사용 모델**: EfficientNetB0 (전이학습)
- **데이터셋**: 데이터셋 미확보 (Imagen3를 통해서 확보중)

## 디렉토리 구조

```
phone_damage_detection/
├── models/                      # 학습된 모델 저장 디렉토리
│   └── phone_damage_EfficientNetB0.h5  # 학습된 모델 파일
├── phone_classification-2/      # 다운로드된 데이터셋
│   ├── train/                   # 학습 데이터
|   |   ├── back_broken/         # 핸드폰 뒷면 파손 이미지
|   |   ├── back_normal/         # 핸드폰 뒷면 이미지
|   |   ├── front_broken/        # 핸드폰 앞면 파손 이미지
|   |   ├── front_normal/        # 핸드폰 앞면 이미지
|   |   └── lines/               # 핸드폰 금간 이미지 (생활 기스 이미지)
│   ├── valid/                   # 검증 데이터 /하위 디렉토리 train과 같음
│   └── test/                    # 테스트 데이터 /하위 디렉토리 train과 같음
├── results/                     # 테스트 결과 이미지
├── download_dataset.py          # 데이터셋 다운로드 스크립트
├── preprocess_data.py           # 데이터 전처리 스크립트
├── model_architecture.py        # 모델 아키텍처 구현 스크립트
├── train_evaluate_model.py      # 모델 학습 및 평가 스크립트
├── prediction_interface.py      # GUI 예측 인터페이스
├── test_model.py                # 모델 테스트 스크립트
└── README.md                    # 프로젝트 설명 문서
```

## 설치 및 실행 방법

### 필수 라이브러리 설치

```bash
pip3 install tensorflow keras matplotlib pandas numpy pillow scikit-learn roboflow seaborn opencv-python
```

### 데이터셋 다운로드

```bash
python3 download_dataset.py
```

### 데이터 전처리

```bash
python3 preprocess_data.py
```

### 모델 아키텍처 구현

```bash
python3 model_architecture.py
```

### 모델 학습 및 평가

```bash
python3 train_evaluate_model.py
```

### 예측 인터페이스 실행

```bash
python3 prediction_interface.py
```

### 모델 테스트

```bash
python3 test_model.py --image [이미지_경로]
```

## 모델 성능

테스트 미실시

### 등급별 성능

테스트 미실시

## 개선 방향

1. **더 많은 데이터 수집**: 각 등급별로 더 많은 이미지 데이터를 수집하여 학습
2. **데이터 증강 기법 개선**: 더 다양한 데이터 증강 기법을 적용하여 데이터셋 확장
3. **모델 아키텍처 최적화**: 다른 모델 아키텍처 시도 또는 하이퍼파라미터 튜닝
4. **앙상블 기법 적용**: 여러 모델의 예측을 결합하여 성능 향상
5. **클래스 불균형 해결**: 클래스 가중치 조정 또는 언더/오버 샘플링 기법 적용

## 참고 사항

- 현재 모델의 정확도는 제한적이므로 결과를 참고용으로만 사용해주세요.
- 더 정확한 분류를 위해서는 추가적인 데이터 수집과 모델 개선이 필요합니다.
