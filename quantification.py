import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# ✅ 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows 기본 한글 폰트
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')    # macOS
else:
    plt.rc('font', family='NanumGothic')    # Linux 등
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# ✅ YOLOv8 학습 결과가 저장된 상위 디렉토리
BASE_PROJECT_DIR = './carparts_dataset/models/yolov8n'  # 사용자 환경에 맞게 수정

# ✅ 별칭 → 실제 YOLOv8 CSV 컬럼 이름 매핑
METRIC_ALIASES = {
    'mAP50'      : ['metrics/mAP50(B)', 'metrics/mAP_0.5'],
    'mAP50-95'   : ['metrics/mAP50-95(B)', 'metrics/mAP_0.5:0.95'],
    'precision'  : ['metrics/precision(B)', 'metrics/precision'],
    'recall'     : ['metrics/recall(B)', 'metrics/recall'],
    'box_loss'   : ['val/box_loss', 'metrics/box_loss'],
    'cls_loss'   : ['val/cls_loss', 'metrics/cls_loss'],
    'dfl_loss'   : ['val/dfl_loss', 'metrics/dfl_loss'],
}

def resolve_metric(df_cols, metric_key):
    if metric_key in df_cols:
        return metric_key
    if metric_key in METRIC_ALIASES:
        for candidate in METRIC_ALIASES[metric_key]:
            if candidate in df_cols:
                return candidate
    return None

def plot_multiple_metrics(base_dir, metrics, runs=None):
    """
    여러 YOLOv8 run의 여러 지표를 한 번에 시각화 + 수치 요약 출력
    """
    if runs is None:
        runs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('fine_tune')]

    if not runs:
        print("❌ 분석할 run 디렉토리를 찾을 수 없습니다.")
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)

    if n_metrics == 1:
        axes = [axes]

    print("📊 Metric 볼륨 요약")
    print("=" * 90)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True)
        plotted = False

        print(f"\n📌 Metric: {metric}")
        print("-" * 90)
        print(f"{'Run':<20} {'최고값':>8} {'최저값':>8} {'std':>8} {'range':>8} {'last':>8} {'best@':>8}")

        for run in runs:
            csv_path = os.path.join(base_dir, run, 'results.csv')
            if not os.path.exists(csv_path):
                print(f"⚠️ {run}: results.csv 없음")
                continue

            df = pd.read_csv(csv_path)
            col = resolve_metric(df.columns, metric)
            if col is None:
                print(f"⚠️ {run}: '{metric}' 지표 없음. 컬럼 목록: {df.columns.tolist()}")
                continue

            y = df[col]
            ax.plot(df.index + 1, y, label=run)
            plotted = True

            # ▶️ 수치 요약 계산
            best_value = y.max() if 'loss' not in metric else y.min()
            worst_value = y.min() if 'loss' not in metric else y.max()
            std = y.std()
            rng = y.max() - y.min()
            last = y.iloc[-1]
            best_epoch = y.idxmax() + 1 if 'loss' not in metric else y.idxmin() + 1

            print(f"{run:<20} {best_value:8.4f} {worst_value:8.4f} {std:8.4f} {rng:8.4f} {last:8.4f} {best_epoch:8}")

        if i == n_metrics - 1:
            ax.set_xlabel('Epoch')
        if plotted:
            ax.legend()

    plt.suptitle("YOLOv8 학습 지표 시각화 + 수치 요약", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ✅ 실행 예시
plot_multiple_metrics(
    base_dir=BASE_PROJECT_DIR,
    metrics=['mAP50', 'mAP50-95', 'precision', 'box_loss']
)
