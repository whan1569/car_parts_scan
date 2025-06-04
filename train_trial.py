import os
import glob
import torch
from ultralytics import YOLO

def find_latest_best(project_dir):
    """
    프로젝트 디렉토리 내에서 가장 최근 수정된 best.pt 경로를 반환.
    없으면 None 반환.
    """
    pattern = os.path.join(project_dir, '**', 'best.pt')
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    # 가장 최근 수정된 best.pt 선택
    return max(candidates, key=os.path.getmtime)

def main():
    # ✅ 설정
    MODEL_NAME = 'yolov8n'
    DATA_YAML_PATH = os.path.join(os.getcwd(), 'carparts_dataset', 'carparts-seg', 'data.yaml')
    PROJECT_DIR = os.path.join(os.getcwd(), 'carparts_dataset', 'models', MODEL_NAME)
    RUN_NAME = 'baseline_run'
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ✅ 이전 best.pt 탐색
    checkpoint_path = find_latest_best(PROJECT_DIR)
    if checkpoint_path:
        print(f"🔁 가장 최근 best.pt에서 이어서 학습: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print(f"🚀 사전학습 모델 {MODEL_NAME}.pt에서 새로 시작")
        model = YOLO(f"{MODEL_NAME}.pt")

    # ✅ 학습 시작
    model.train(
        data=DATA_YAML_PATH,
        epochs=500,
        patience=10,
        batch=16,
        imgsz=640,
        project=PROJECT_DIR,
        name=RUN_NAME,
        device=DEVICE,
        verbose=True
    )

if __name__ == "__main__":
    main()
