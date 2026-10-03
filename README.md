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
| `remap_ciciot2023_labels.py` | CICIoT2023 원본 34개 세부 레이블을 논문이 명시한 6-class(Benign/DDoS/DoS/MITM/Mirai/Recon, 기본값) 또는 데이터셋 공식 8-class(Benign + DDoS/DoS/Mirai/Recon/Spoofing/Web/BruteForce, `--scheme 8class`)로 매핑 — **CICIoT2023은 항상 이걸 거친 뒤 사용** (아래 3절 참고) |
| `filter_ciciot2023_binary.py` | 팀 연구계획서가 요구하는 Benign vs DDoS 이진분류용 데이터 생성 — 그 외 공격 유형(DoS/Mirai/MITM/Recon/Web/BruteForce) 행은 제거(다른 클래스로 합치지 않음) |
| `sample_dataset.py` | 클래스당 최대 N개로 상한을 두는 메모리 절약형 샘플링 (전체 클래스 유지) |
| `feature_selection.py` | 연구계획서 2.3절 Feature Selection: Random Forest 기반 importance + SHAP 기반 importance 산출, 두 방식 비교(Spearman correlation, Top-k overlap), Top-k feature subset CSV 생성 |
| `summarize_results.py` | `results/` 아래 여러 `train.py` 실행 결과(`*_summary.json` + `*_per_class_metrics.csv`)를 accuracy 기준 정렬된 표 하나로 모아서 비교 (baseline vs Top-k 실험들을 한 번에 조회) |

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

### CICIoT2023은 항상 6-class로 매핑해서 사용 (34-class 금지)

CICIoT2023 원본은 BenignTraffic + 33개 세부 공격 레이블(34-class)입니다.
**이 34개를 그대로 학습에 쓰지 않습니다** — `DDoS-SYN_Flood` vs
`DDoS-RSTFINFlood` vs `DDoS-ACK_Fragmentation`처럼 서로 거의 구분이 안
되는 세부 유형들까지 다 구분하게 만드는 데다, 유형별 샘플 수 편차가 너무
커서(수백 개~80만 개) 소수 클래스 recall이 0에 가깝게 무너지고 weighted
accuracy가 크게 낮아집니다(실측: baseline 91%대, 반면 AUC는 0.99+로 모델
자체의 판별력은 충분함 — 즉 세분화·불균형 문제이지 모델/코드 문제가
아님).

처음에는 원 논문이 CICIoT2023에서 몇 개 클래스를 썼는지 안 밝혔다고
판단해서 데이터셋 공식 8-class(Benign + DDoS/DoS/Mirai/Recon/Spoofing/
Web/BruteForce)를 기본값으로 썼었는데, **논문 본문(Section 4.1.3)과
Table 6을 직접 확인해보니 실제로는 6-class(Benign, DDoS, DoS, MITM,
Mirai, Recon)라고 명시되어 있었습니다.** 8-class 대비 Web·BruteForce
(샘플 수가 가장 적고 오분류되기 쉬운 두 클래스)가 빠지고, Spoofing이
"MITM"이라는 이름으로 통합되어 있습니다. **따라서 기본값을 6-class로
변경했습니다.** (다만 "MITM"이 원본 34-class 중 `MITM-ArpSpoofing`만
포함하는지 `DNS_Spoofing`까지 포함하는지는 논문에 명시되어 있지 않아,
현재는 둘 다 합쳐서 매핑 — 저자 확인이 필요한 부분.)

**CICIoT2023 관련 스크립트를 돌리기 전, 항상 먼저 6-class로 변환하세요:**

```bash
python remap_ciciot2023_labels.py --in data/CICIOT23/train/train.csv \
    --out data/CICIOT23/train/train_6class.csv --label_col label
```

데이터셋 공식 8-class와 비교해보고 싶다면 `--scheme 8class`를 추가하세요:

```bash
python remap_ciciot2023_labels.py --in data/CICIOT23/train/train.csv \
    --out data/CICIOT23/train/train_8class.csv --label_col label --scheme 8class
```

그 다음 `train.py`/`feature_selection.py`/`sample_dataset.py`에는 원본
`train.csv`가 아니라 이 `train_6class.csv`(또는 `train_8class.csv`)를
`--csv`로 넘기세요.

## 재현상 확인이 필요한 사항 (ambiguities)

논문이 명시하지 않은 부분은 기본값을 **가장 문자 그대로에 가까운 쪽**으로
맞춰뒀습니다. 성능을 더 끌어올리는 실험을 하고 싶을 때만 아래 플래그로
의도적으로 벗어날 수 있게 열어둔 구조입니다.

- `model.py`: 논문 Eq. (3)은 활성화 함수 없는 순수 affine 변환으로
  embedding을 정의하지만, 3.2절 본문은 "hidden layers"(복수형)에 ReLU를
  적용한다고 서술 — **2026-09-28 교신저자 Jehad Ali 교수님이 이메일로
  직접 확인**: "두 hidden layer(128, 64) 다 ReLU를 씀". 더 이상
  ambiguity가 아니라 확정된 사실 — **기본값을
  `embedding_activation=True`로 변경**. Eq. (3) 문자 그대로(ReLU 없음)
  재현하려면 `train.py --no_embedding_activation`.
- `model.py`: Figure 1은 두 MLP block을 각각 "Dense+ReLU → LayerNorm"으로
  그려놨는데, Eq. (2)/(3)이나 본문 어디에도 LayerNorm 언급이 없음 —
  그림에만 존재하는 요소. Figure 1이 실제 구현을 더 정확히 반영했을
  가능성이 높다고 보고(lr=0.003 + scheduler 없음인데도 논문의 학습
  곡선이 저희 재현보다 훨씬 안정적인 것과도 부합) **기본값을
  `layer_norm=True`로 둠**. 수식/본문을 문자 그대로 따르려면
  `train.py --no_layer_norm`.
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
  진동하는 경향이 있음 — 같은 설정으로 재실행해도 마지막 epoch 정확도가
  실행마다 크게 달라짐(실측: 84~95%대로 요동). **기본값을
  `select_best_epoch=True`로 둠**: train을 다시 train/validation으로
  나누고, validation accuracy가 가장 높았던 epoch의 가중치를 최종
  평가에 씀(`--val_size`로 비율 조절). test set은 선택에 전혀 관여하지
  않으므로 낙관 편향(data leakage)은 없음 — 다만 이 checkpoint-선택
  절차 자체가 논문 Section 4.1에는 없는 내용이라는 점은 유의. Section
  4.1을 문자 그대로 재현하려면(마지막 100번째 epoch 가중치 그대로 평가)
  `--no_select_best_epoch`을 사용.
- **`--lr_scheduler` (기본값 False, 재현이 아니라 개선 실험용 opt-in)**:
  위 출렁임 문제의 근본 원인(고정 lr=0.003)을 best-epoch 선택으로
  우회하는 대신, cosine annealing으로 lr 자체를 100 epoch에 걸쳐
  서서히 줄여서 진동을 줄일 수 있는지 시험하는 옵션. Table 3에 없는
  내용이라 기본값은 꺼져 있고, baseline(`--lr_scheduler` 없음)과
  개선 실험(`--lr_scheduler`)을 나란히 비교하는 용도.
- CICIoT2023의 실제 클래스 개수: 처음에는 논문에 전혀 언급이 없다고
  판단해서 데이터셋 공식 8-class를 기본값으로 썼었으나, **논문 본문
  Section 4.1.3과 Table 6을 직접 확인한 결과 "Benign, DDoS, DoS, MITM,
  Mirai, Recon" 6-class로 명시되어 있었음** — 이 항목은 더 이상
  ambiguity가 아니라 논문에 명시된 사실. **2026-09-28 교신저자 Jehad Ali
  교수님이 이메일로도 직접 재확인**("The reported 98.45% accuracy on
  CICIoT2023 corresponds to the six-class setting"). **기본값을
  6-class로 변경함** (`remap_ciciot2023_labels.py`, 3절 참고). 다만
  "MITM"이 원본 34-class 중 정확히 어떤 서브타입(`MITM-ArpSpoofing`만인지
  `DNS_Spoofing`까지 포함인지)을 가리키는지는 여전히 논문/답장 둘 다에
  없어 둘 다 합쳐서 매핑 중 — 이 부분만 남은 ambiguity.
- 34-class로 확장 실험을 할 경우, 저자 권고: imbalance-handling 기법을
  적용하고 accuracy 외에 macro-averaged precision/recall/F1도 함께
  보고할 것 (2026-09-28 이메일 답변). 아직 코드에 반영되지 않은 향후
  과제.
