from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from app import ImageTransformSSL, cached_dataset, plot_history, run_mae, run_rotation, to_pil


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
REPORT = OUT_DIR / "self_supervised_vision_report.pdf"


def chinese_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            return fm.FontProperties(fname=str(path))
    return fm.FontProperties()


FONT = chinese_font()


def add_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69), dpi=140)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontproperties=FONT, fontsize=20, weight="bold")
    y = 0.88
    for line in lines:
        if line == "":
            y -= 0.03
            continue
        fig.text(0.08, y, line, fontproperties=FONT, fontsize=11, va="top", wrap=True)
        y -= 0.042
    pdf.savefig(fig)
    plt.close(fig)


def add_mae_page(pdf):
    low = run_mae(0.35, 160, 5)
    high = run_mae(0.65, 160, 18)
    fig = plt.figure(figsize=(11.69, 8.27), dpi=140)
    fig.suptitle("MAE 遮挡重建：输入、遮挡、输出与 loss 对比", fontproperties=FONT, fontsize=17)
    axes = []
    for i in range(2):
        for j in range(3):
            axes.append(fig.add_subplot(2, 4, i * 4 + j + 1))
    curve_ax = fig.add_subplot(1, 4, 4)

    rows = [(low, "mask=0.35"), (high, "mask=0.65")]
    for row, (result, label) in enumerate(rows):
        for col, (name, image) in enumerate(
            [("Input", result.original_image), ("Masked", result.masked_image), ("Output", result.recon_image)]
        ):
            ax = axes[row * 3 + col]
            ax.imshow(to_pil(image))
            ax.set_title(f"{label} {name}", fontproperties=FONT, fontsize=10)
            ax.axis("off")
    curve_ax.plot(low.losses, label="mask=0.35", linewidth=2)
    curve_ax.plot(high.losses, label="mask=0.65", linewidth=2)
    curve_ax.set_title("Loss", fontproperties=FONT)
    curve_ax.set_xlabel("step")
    curve_ax.grid(True, alpha=0.25)
    curve_ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def add_rotation_page(pdf):
    loss, acc = run_rotation(160, 9)
    images = cached_dataset(8, 42)
    transform = ImageTransformSSL()
    preview = images[1]
    fig = plt.figure(figsize=(11.69, 8.27), dpi=140)
    fig.suptitle("旋转预测：图像变换类自监督任务与准确率变化", fontproperties=FONT, fontsize=17)
    for k in range(4):
        ax = fig.add_subplot(2, 4, k + 1)
        ax.imshow(to_pil(transform.rotate(preview, k)))
        ax.set_title(f"{k * 90} deg", fontsize=10)
        ax.axis("off")
    ax_loss = fig.add_subplot(2, 2, 3)
    ax_acc = fig.add_subplot(2, 2, 4)
    ax_loss.plot(loss, linewidth=2)
    ax_loss.set_title("Cross entropy loss", fontproperties=FONT)
    ax_loss.set_xlabel("step")
    ax_loss.grid(True, alpha=0.25)
    ax_acc.plot(acc, color="#2b8a3e", linewidth=2)
    ax_acc.set_title("Accuracy", fontproperties=FONT)
    ax_acc.set_xlabel("step")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    prompt = [
        "Prompt：",
        "Vibe Coding",
        "1. 实现一个图像变换类自监督学习示例，如旋转预测、拼图重排、图像补全或颜色化；",
        "2. 实现一个 MAE 或 SimCLR 的简化示例，如图像遮挡重建，或基于数据增强的正负样本对比学习；",
        "3. 可视化输入图像、变换/遮挡后的图像、模型输出结果，并简单展示 loss 或准确率变化；",
        "4. 简单对比两种设置的效果，如不同遮挡比例、不同数据增强方式、训练前后效果对比；",
        "5. 实现成一个桌面app或web应用的形式，需要有交互操作；",
        "6. 提交 PDF 实验报告、核心源码 py 文件、可外链访问的 url（可选）。",
        "",
        "使用的 Agent / LLM：Codex GPT-5。",
    ]
    summary = [
        "实验小结：",
        "本项目实现了两个简化自监督视觉任务。",
        "MAE 部分把 32x32 图像切成 4x4 patch，随机遮挡一部分 patch，再训练一个 NumPy 线性 patch decoder 预测被遮挡 patch 的 RGB 均值。",
        "旋转预测部分将同一图像旋转 0/90/180/270 度，用手工统计特征和 softmax 分类器预测旋转角度。",
        "交互式 Streamlit 页面支持调节随机种子、训练步数、遮挡比例和对比遮挡比例，并实时展示输入、变换后图像、模型输出和曲线。",
        "对比结果通常显示：较高遮挡比例的信息更少，重建更困难，loss 更高或收敛更慢；旋转预测随着训练推进，准确率明显高于训练初期。",
        "",
        "核心源码：app.py",
        "本地运行：streamlit run app.py",
    ]

    with PdfPages(REPORT) as pdf:
        add_text_page(pdf, "自监督视觉实验报告", prompt)
        add_mae_page(pdf)
        add_rotation_page(pdf)
        add_text_page(pdf, "小结与提交信息", summary)
    print(REPORT)


if __name__ == "__main__":
    main()
