import os
import torch
import time
import optuna
import pandas as pd
from ultralytics import YOLO

MODEL_NAME = 'yolov8n'
DATA_YAML_PATH = os.path.join(os.getcwd(), 'carparts_dataset', 'carparts-seg', 'data.yaml')
BASE_PROJECT_DIR = os.path.join(os.getcwd(), 'carparts_dataset', 'models', MODEL_NAME)
EPOCHS = 1000
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_next_run_name(base_dir, base_name='fine_tune'):
    if not os.path.exists(base_dir):
        return base_name
    existing = [d for d in os.listdir(base_dir) if d.startswith(base_name)]
    nums = [int(d.replace(base_name, '').strip('_')) for d in existing if d.replace(base_name, '').strip('_').isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"{base_name}_{next_num}"

def find_best_checkpoint(base_dir):
    if not os.path.exists(base_dir):
        return None
    for subdir in sorted(os.listdir(base_dir), reverse=True):
        weights_path = os.path.join(base_dir, subdir, "weights", "best.pt")
        if os.path.exists(weights_path):
            return weights_path
    return None

def objective(trial):
    lr = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    batch = trial.suggest_categorical("batch", [8, 16, 32])

    run_name = get_next_run_name(BASE_PROJECT_DIR)
    run_dir = os.path.join(BASE_PROJECT_DIR, run_name)

    checkpoint = find_best_checkpoint(BASE_PROJECT_DIR)
    if checkpoint:
        print(f"🔁 이전 best.pt로부터 이어서 학습: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        print(f"🚀 사전학습 모델 yolov8n.pt에서 새로 시작")
        model = YOLO(f"{MODEL_NAME}.pt")

    # 학습 시작 (최대 EPOCHS)
    # verbose=True로 출력 확인 가능
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=batch,
        lr0=lr,
        project=BASE_PROJECT_DIR,
        name=run_name,
        device=DEVICE,
        verbose=False,
    )

    # EarlyStopping 및 Optuna pruning 수동 처리
    patience = 5
    delta = 0.005
    best_map = None
    counter = 0

    results_csv = os.path.join(run_dir, "results.csv")

    for epoch in range(EPOCHS):
        # 결과 파일 존재 체크
        if not os.path.exists(results_csv):
            time.sleep(2)
            continue

        df = pd.read_csv(results_csv)
        if len(df) <= epoch:
            time.sleep(2)
            continue

        # mAP50 컬럼 확인
        map50 = None
        if "metrics/val/box/mAP50" in df.columns:
            map50 = df["metrics/val/box/mAP50"].iloc[epoch]
        elif "val/box/mAP50" in df.columns:
            map50 = df["val/box/mAP50"].iloc[epoch]
        else:
            print("mAP50 컬럼을 찾을 수 없습니다.")
            break

        trial.report(map50, epoch)
        if trial.should_prune():
            print(f"🔥 Optuna pruning triggered at epoch {epoch}, mAP50={map50:.4f}")
            raise optuna.exceptions.TrialPruned()

        if best_map is None or map50 > best_map + delta:
            best_map = map50
            counter = 0
            print(f"📈 Epoch {epoch}: mAP50 향상됨: {map50:.4f}")
        else:
            counter += 1
            print(f"😐 Epoch {epoch}: 개선 없음 {counter}/{patience}")
            if counter >= patience:
                print(f"🛑 EarlyStopping triggered at epoch {epoch}")
                break

        time.sleep(2)  # 너무 자주 체크하지 않도록 딜레이

    return best_map or 0

def run_optuna():
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=30)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    trial_no_improve_count = 0
    best_trial_score = None

    N_TRIALS = 20

    for i in range(N_TRIALS):
        print(f"\n▶▶▶ Trial {i+1}/{N_TRIALS} 시작")
        try:
            study.optimize(objective, n_trials=1)
        except optuna.exceptions.TrialPruned:
            print("⚠️ 트라이얼 조기 종료됨.")

        current_best = study.best_value
        print(f"현재 최고 점수: {current_best:.4f}")

        if best_trial_score is None or current_best > best_trial_score + 0.005:
            best_trial_score = current_best
            trial_no_improve_count = 0
            print("✨ 최고 점수 개선됨, 조기 종료 카운터 초기화")
        else:
            trial_no_improve_count += 1
            print(f"😐 최고 점수 개선 없음: {trial_no_improve_count}/3")

        if trial_no_improve_count >= 3:
            print("🛑 트라이얼 단위 조기 종료 조건 충족: 3회 연속 개선 없음")
            break

    print("✅ 최적 하이퍼파라미터:", study.best_params)
    print(f"최고 mAP: {best_trial_score:.4f}")

if __name__ == "__main__":
    run_optuna()
