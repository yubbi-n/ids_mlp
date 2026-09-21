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

## 재현상 확인이 필요한 사항 (ambiguities)

- `model.py`: 논문 Eq. (3)은 활성화 함수 없는 순수 affine 변환으로
  embedding을 정의하지만, 3.2절 본문은 "hidden layers"(복수형)에 ReLU를
  적용한다고 서술 — 현재는 Eq. (3)을 문자 그대로 따르되
  `embedding_activation` 플래그로 전환 가능하게 구현.
- `preprocess.py`: Min-Max 정규화를 train/test split 이전(Algorithm 1
  순서)에 전체 데이터에 fit할지, split 이후 train에만 fit할지 불명확 —
  data leakage 방지를 위해 train에만 fit하도록 구현(`train.py`,
  `feature_selection.py` 공통).
- `train.py`: FLOPs/MACs 계산에 사용한 프로파일링 도구(thop/ptflops/fvcore
  등)가 논문에 명시되어 있지 않음 — 현재는 weight 곱셈 기준 단순 해석적
  추정치를 사용.
- 80/20 split의 stratify 여부, random seed 값도 논문에 명시되어 있지
  않음 — 현재 구현은 `stratify=y`, `seed=42`를 기본값으로 사용.
- 논문이 학습을 한 번만 돌려서 보고했는지, 여러 번 돌려 평균/표준편차를
  낸 것인지 불명확 — Section 4.1에는 "80/20 split"만 언급되고 반복 실행
  여부는 명시되어 있지 않음. 본 재현은 기본적으로 단일 실행 기준.
- `train.py`는 lr=0.003이 이 모델/데이터 규모엔 다소 높아 학습 곡선이
  진동하는 경향이 있어(README 재현 로그 참고), epoch 100의 값을 그대로
  최종 성능으로 쓰면 "우연히 어느 epoch에서 끝났는지"에 좌우되는 문제가
  있었음. 이를 보완하기 위해 train을 다시 train/validation으로 나누고
  (`--val_size`, 기본 0.1), **validation accuracy가 가장 높았던 epoch의
  가중치**를 최종 평가(test set)에 사용하도록 변경 — test set은 모델
  선택에 전혀 관여하지 않아 결과가 낙관적으로 치우치지 않음. 논문의
  문자 그대로의 80/20 split을 원하면 `--val_size 0`으로 이전 동작(마지막
  epoch 기준, test set으로 best-epoch 선택)으로 되돌릴 수 있음.
