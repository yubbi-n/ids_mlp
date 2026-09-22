# ids_mlp

「Lightweight MLP-Based Feature Extraction with Linear Classifier for
Intrusion Detection System in Internet of Things」(Chandroth & Ali,
*Electronics* 2026, 15(8), 1604) 재현 코드 + 연구계획서(특징 선택 및
경량화 기법 기반 Edge AI IoT 침입 탐지 및 자원 효율 분석) 1단계
(Feature Selection) 실험 코드.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `model.py` | 논문 3.2-3.3절 구조 재현 (2-layer MLP embedding + linear softmax classifier) |
| `preprocess.py` | 논문 3.1절 전처리 파이프라인 (결측치/비유한값 제거 → 레이블 인코딩 → 원-핫 인코딩 → Min-Max 정규화) |
| `train.py` | 논문 4절 학습/평가 (AdamW, lr=0.003, wd=1e-4, 100 epoch, batch 128, 80/20 split, Accuracy/Precision/Recall/F1/AUC, params/FLOPs/model size/시간 측정) |
| `remap_nslkdd_labels.py` | NSL-KDD 원본 공격 레이블을 Normal/DoS/Probe/R2L/U2R 5-class로 매핑 |
| `remap_ciciot2023_labels.py` | CICIoT2023 원본 34개 세부 레이블을 공식 8-class(Benign + DDoS/DoS/Mirai/Recon/Spoofing/Web/BruteForce)로 매핑 — **CICIoT2023은 항상 이걸 거친 뒤 사용** (아래 3절 참고) |
| `sample_dataset.py` | 클래스당 최대 N개로 상한을 두는 메모리 절약형 샘플링 (전체 클래스 유지) |
| `feature_selection.py` | 연구계획서 2.3절 Feature Selection: Random Forest 기반 importance + SHAP 기반 importance 산출, 두 방식 비교(Spearman correlation, Top-k overlap), Top-k feature subset CSV 생성 |

## 1. 가상환경 설정 (conda)

```bash
conda env create -f environment.yml
conda activate ids_mlp
```

- 기본값은 CPU 전용 PyTorch입니다. GPU를 사용하려면 `environment.yml`에서
  `cpuonly` 줄을 지우고, [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)에서
  본인의 CUDA 드라이버 버전에 맞는 `pytorch-cuda=<version>` 패키지로 바꿔서
  다시 `conda env create`를 실행하세요.
- 환경을 이미 만든 뒤 패키지 목록만 바뀐 경우: `conda env update -f environment.yml --prune`

## 2. 사용법

### 2.1 기준 모델 학습 (Baseline MLP)

```bash
python train.py --csv data/NSL-KDD_5class.csv --label_col label \
    --categorical_cols protocol_type service flag \
    --name NSL-KDD --output_dir results
```

### 2.2 Feature Selection (Random Forest / SHAP)

전체 feature로 baseline 성능을 낸 뒤, 아래 명령으로 feature importance를
산출합니다. Section 2.3-(4)의 data leakage 방지 원칙에 따라 importance는
train.py와 **동일한 시드(`--seed`)·동일한 split 비율(`--test_size`)로
분할한 train set에서만** 계산됩니다.

```bash
# SHAP을 Random Forest에 대해 계산 (기본값)
python feature_selection.py --csv data/NSL-KDD_5class.csv --label_col label \
    --categorical_cols protocol_type service flag \
    --name NSL-KDD --topk 10 20 30 --output_dir results

# SHAP을 MLP(baseline 모델)에 대해 계산하고 싶은 경우
python feature_selection.py --csv data/NSL-KDD_5class.csv --label_col label \
    --categorical_cols protocol_type service flag \
    --name NSL-KDD --shap_model mlp --mlp_epochs 30 --output_dir results
```

출력물 (`--output_dir` 아래):

- `<name>_feature_ranking.csv` — feature별 RF importance / SHAP importance / 두 랭킹
- `<name>_feature_importance.png` — RF vs SHAP top-20 feature bar chart
- `<name>_rf_top{k}.csv`, `<name>_shap_top{k}.csv` — 각 방법의 Top-k feature만
  남긴 CSV (label 컬럼 포함). 이 파일을 그대로 `train.py --csv`에 넣으면
  Baseline(전체 feature) vs FS-k(선정 feature) 성능을 바로 비교할 수 있습니다.

콘솔에는 두 importance 랭킹의 Spearman correlation과 Top-k overlap 비율이
함께 출력되어, RF 기반과 SHAP 기반 selection이 얼마나 일치하는지 확인할 수
있습니다 (연구계획서 2.3-(2) "RF 기반 importance와 SHAP 기반 importance의
비교").

## 3. 데이터셋

연구계획서(2.1절) 기준 대상 데이터셋: **CICIoT2023**, **CICIDS2017**,
**UNSW-NB15**. 각 데이터셋의 실험상 역할(주 실험/검증/일반화 검증)은 초기
실험 결과를 보고 결정하며, 세 데이터셋을 동일 비중으로 모두 사용한다고
현재 확정되어 있지는 않습니다.

NSL-KDD를 사용하는 경우 `remap_nslkdd_labels.py`로 원본 공격 레이블을
5-class(Normal/DoS/Probe/R2L/U2R)로 변환한 뒤 `train.py --label_col`에
매핑된 컬럼명을 지정하세요.

### CICIoT2023은 항상 8-class로 매핑해서 사용 (34-class 금지)

CICIoT2023 원본은 BenignTraffic + 33개 세부 공격 레이블(34-class)입니다.
**이 34개를 그대로 학습에 쓰지 않습니다** — `DDoS-SYN_Flood` vs
`DDoS-RSTFINFlood` vs `DDoS-ACK_Fragmentation`처럼 서로 거의 구분이 안
되는 세부 유형들까지 다 구분하게 만드는 데다, 유형별 샘플 수 편차가 너무
커서(수백 개~80만 개) 소수 클래스 recall이 0에 가깝게 무너지고 weighted
accuracy가 크게 낮아집니다(실측: baseline 91%대, 반면 AUC는 0.99+로 모델
자체의 판별력은 충분함 — 즉 세분화·불균형 문제이지 모델/코드 문제가
아님). 원 논문도 CICIoT2023에서 몇 개 클래스를 썼는지 명시하지 않았고,
대부분의 CICIoT2023 관련 논문은 공식 8-class(Benign + DDoS/DoS/Mirai/
Recon/Spoofing/Web/BruteForce) 기준으로 성능을 보고합니다.

**따라서 CICIoT2023 관련 스크립트를 돌리기 전, 항상 먼저 8-class로 변환하세요:**

```bash
python remap_ciciot2023_labels.py --in data/CICIOT23/train/train.csv \
    --out data/CICIOT23/train/train_8class.csv --label_col label
```

그 다음 `train.py`/`feature_selection.py`/`sample_dataset.py`에는 원본
`train.csv`가 아니라 이 `train_8class.csv`를 `--csv`로 넘기세요.

## 재현상 확인이 필요한 사항 (ambiguities)

논문이 명시하지 않은 부분은 기본값을 **가장 문자 그대로에 가까운 쪽**으로
맞춰뒀습니다. 성능을 더 끌어올리는 실험을 하고 싶을 때만 아래 플래그로
의도적으로 벗어날 수 있게 열어둔 구조입니다.

- `model.py`: 논문 Eq. (3)은 활성화 함수 없는 순수 affine 변환으로
  embedding을 정의하지만, 3.2절 본문은 "hidden layers"(복수형)에 ReLU를
  적용한다고 서술 — **기본값은 Eq. (3) 문자 그대로(ReLU 없음,
  `embedding_activation=False`)**. `train.py --embedding_activation`으로
  3.2절 해석(ReLU 있음)을 켤 수 있음.
- `preprocess.py`/`train.py`: Min-Max 정규화를 train/test split 이전
  (Algorithm 1 순서)에 전체 데이터에 fit할지, split 이후 train에만
  fit할지 논문상 불명확 — **기본값은 Algorithm 1 순서 그대로 전체
  데이터에 fit(`scale_before_split=True`)**. `train.py
  --no_scale_before_split`으로 data leakage 없는 train-only fit으로
  바꿀 수 있음 (`feature_selection.py`는 research plan 2.3-(4)의 별도
  요구사항에 따라 항상 train-only fit).
- `train.py`: FLOPs/MACs 계산에 사용한 프로파일링 도구(thop/ptflops/fvcore
  등)가 논문에 명시되어 있지 않음 — 현재는 weight 곱셈 기준 단순 해석적
  추정치를 사용.
- 80/20 split의 stratify 여부, random seed 값도 논문에 명시되어 있지
  않음 — 대체할 논문 명시값 자체가 없어 임의로 `stratify=y`, `seed=42`를
  기본값으로 사용 (되돌릴 "문자 그대로"의 기준이 존재하지 않는 항목).
- 논문이 학습을 한 번만 돌려서 보고했는지, 여러 번 돌려 평균/표준편차를
  낸 것인지 불명확 — Section 4.1에는 "80/20 split"만 언급되고 반복 실행
  여부는 명시되어 있지 않음. 본 재현은 기본적으로 단일 실행 기준.
- `train.py`는 lr=0.003이 이 모델/데이터 규모엔 다소 높아 학습 곡선이
  진동하는 경향이 있음. **기본값은 Section 4.1이 실제로 서술한 그대로
  "100 epoch 학습 후 그 시점 가중치로 평가"** — best-checkpoint 선택
  과정은 논문에 전혀 언급되지 않으므로 기본으로 켜지 않음
  (`select_best_epoch=False`). 학습 곡선이 출렁여 마지막 epoch 값이
  우연에 좌우되는 게 걱정되면 `--select_best_epoch`(+ `--val_size`)로
  train을 다시 train/validation으로 나누고 validation accuracy가 가장
  높았던 epoch의 가중치를 최종 평가에 쓰도록 켤 수 있음(test set은
  선택에 관여하지 않아 낙관 편향은 없지만, 논문에 없는 절차라는 점은
  동일).
- CICIoT2023의 실제 클래스 개수(34-class 원본 vs 다른 CICIoT2023
  논문들이 흔히 쓰는 8-class 카테고리)는 본 논문에 전혀 언급되어 있지
  않음 — 34-class 그대로 쓰면 클래스 불균형으로 소수 클래스 recall이
  0에 가까워지는 문제가 실측 확인되어(baseline accuracy 91%대, AUC는
  0.99+로 모델 판별력 자체는 정상), **기본값을 8-class로 정함**
  (`remap_ciciot2023_labels.py`, 3절 참고). "논문 미기재 항목"이라는
  점에서 34-class와 8-class 둘 다 확인되지 않은 추정이지만, 8-class가
  실용적으로 더 합리적인 기본값이라 판단.
