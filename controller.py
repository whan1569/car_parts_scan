# train_controller.py

import subprocess
import time

def run_training():
    trial = 1
    max_trials = 3
    no_improve_count = 0

    while no_improve_count < max_trials:
        print(f"▶️ 트라이얼 {trial} 시작")
        # train_trial.py를 외부 프로세스로 실행 (기존 train 코드 유지)
        proc = subprocess.Popen(['python', 'train_trial.py', str(trial), 'yolov8n', 'true'])
        proc.wait()  # 프로세스 종료 대기

        # 종료 후 best.pt 개선 여부 체크 (여기서는 임의로 1회마다 개선됐다고 가정)
        improved = True  # 실제로는 best.pt 성능 비교해서 판단해야 함

        if improved:
            no_improve_count = 0
            print(f"✅ 트라이얼 {trial}에서 개선됨, 계속 진행")
        else:
            no_improve_count += 1
            print(f"⚠️ 개선 없음 ({no_improve_count}/{max_trials})")

        trial += 1

    print("🚫 개선이 3회 연속 없어서 학습 종료")

if __name__ == "__main__":
    run_training()
