# 자동차 부품 세그멘테이션 프로젝트

이 프로젝트는 **자동차 부품 이미지 데이터셋**을 활용하여 **YOLOv8 모델을 파인튜닝(fine-tuning)**하고, 다양한 실험(트라이얼)과 에포크 단위로 자동화된 학습 및 성능 분석을 수행하는 데 목적이 있습니다.

- **자동화된 반복 학습(Trial & Epoch)**: 
  - 한 번의 학습(Trial)마다 여러 에포크(epoch) 동안 모델을 학습합니다.
  - 각 트라이얼(trial)마다 성능이 개선되지 않으면(최대 3회 연속) 자동으로 학습을 종료합니다. (일종의 벡트래킹/backtracking 방식)
  - best.pt(최고 성능 모델) 기준으로 이어서 학습하거나, 새로 시작할 수 있습니다.
- **결과 분석 및 시각화**:
  - 여러 실험(run)의 mAP, precision, loss 등 다양한 지표를 표와 그래프로 한 번에 비교/분석할 수 있습니다.

---

## 폴더 구조

```
car_parts_scan/
├── train_trial.py           # YOLOv8 학습 및 체크포인트 관리 스크립트
├── controller.py            # 반복 학습(Trial) 자동화 컨트롤러
├── quantification.py        # 학습 결과 수치 요약 및 시각화
├── visualization.py         # 학습 지표 시각화
├── yolov8n.pt               # YOLOv8n 사전학습 모델 가중치
├── requirements.txt         # 필요 라이브러리 목록
├── best_load_log.txt        # best.pt 관련 로그
├── gpu_check.py             # GPU 사용 가능 여부 확인 스크립트
├── carparts_dataset/        # 데이터셋 및 모델 결과 폴더
│   ├── carparts-seg/        # 실제 이미지 및 라벨, data.yaml 포함
│   │   ├── train/           # 학습 이미지/라벨
│   │   ├── valid/           # 검증 이미지/라벨
│   │   ├── test/            # 테스트 이미지/라벨
│   │   └── data.yaml        # 데이터셋 구성 및 클래스 정의
│   └── models/              # 학습 결과(run) 및 체크포인트 저장
└── ...
```

## 주요 파일 설명

- **train_trial.py**  
  - YOLOv8 모델을 자동차 부품 데이터셋으로 파인튜닝합니다.
  - 기존 best.pt가 있으면 이어서 학습하며, 없으면 사전학습 가중치로부터 시작합니다.
  - `carparts_dataset/models/yolov8n` 경로에 결과가 저장됩니다.

- **controller.py**  
  - 여러 번의 학습(Trial)을 자동으로 반복 실행합니다.
  - 각 트라이얼마다 성능 개선 여부를 판단하여, 개선이 없을 경우 최대 3회까지 반복 후 자동 종료합니다.
  - 벡트래킹(backtracking)처럼, best.pt 기준으로만 다음 트라이얼을 이어갑니다.

- **quantification.py**  
  - 여러 run의 학습 결과를 수치로 요약하고, 표 형태로 출력합니다.
  - mAP, precision, loss 등 다양한 지표를 한 번에 비교할 수 있습니다.

- **visualization.py**  
  - 여러 run의 학습 지표(mAP, loss 등)를 그래프로 시각화합니다.

- **requirements.txt**  
  - 필수 라이브러리:  
    - ultralytics, torch, pandas, matplotlib, nvidia-ml-py3, optuna

- **gpu_check.py**  
  - GPU 사용 가능 여부를 간단히 확인하는 스크립트입니다.

## 데이터셋 구성

- **carparts_dataset/carparts-seg/data.yaml**  
  - 학습/검증/테스트 이미지 경로와 클래스 정보가 정의되어 있습니다.
  - 총 23개 클래스(예: back_bumper, front_door, wheel 등)가 존재합니다.
  - 데이터셋 다운로드 링크: https://ultralytics.com/assets/carparts-seg.zip

- **train/valid/test**  
  - 각각 학습, 검증, 테스트 이미지와 라벨이 저장되어 있습니다.

## 실행 방법 예시

1. 라이브러리 설치  
   ```
   pip install -r requirements.txt
   ```

2. 학습 실행  
   ```
   python train_trial.py
   ```

3. 반복 학습(Trial) 실행  
   ```
   python controller.py
   ```

4. 학습 결과 시각화/수치 요약  
   ```
   python quantification.py
   python visualization.py
   ``` 