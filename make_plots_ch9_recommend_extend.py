from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures" / "recommend_extend"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_portability_matrix() -> None:
    """
    Conceptual portability difficulty matrix:
      0 = 易, 1 = 中, 2 = 难
    """
    rows = [
        "智能手机/平板",
        "笔记本电脑",
        "智能手表/穿戴",
        "真无线耳机",
        "IoT 传感器节点",
        "无人机/机器人",
    ]
    cols = [
        "SOC/电流可见",
        "主要驱动可见\n(CPU/屏幕/网)",
        "工作模式平稳性",
        "电源架构复杂度",
        "热管理耦合强度",
        "跨设备可比性",
    ]

    # Heuristic scores
    M = np.array(
        [
            [0, 0, 1, 1, 1, 0],  # phone/tablet
            [0, 1, 1, 2, 2, 1],  # laptop
            [1, 1, 2, 1, 2, 2],  # wearable
            [1, 2, 2, 1, 1, 2],  # earbuds
            [1, 2, 2, 2, 1, 2],  # IoT
            [1, 2, 2, 2, 2, 2],  # drone/robot
        ],
        dtype=float,
    )

    fig = plt.figure(figsize=(10.0, 4.2), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=2)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_title("模型框架对不同便携设备的可拓展性：难度矩阵（0易/1中/2难）")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, ["易", "中", "难"][int(M[i, j])], ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1, 2])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "portability_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_aging_concept() -> None:
    """
    Conceptual aging curves and how they affect time-to-empty:
      - capacity fades down
      - internal resistance increases up
    """
    x = np.linspace(0, 1000, 200)  # cycles or equivalent full cycles
    cap = 1.0 - 0.2 * (1 - np.exp(-x / 400))  # saturating fade to ~80%
    rint = 1.0 + 0.6 * (1 - np.exp(-x / 300))  # saturating rise

    fig = plt.figure(figsize=(9.6, 3.6), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(x, cap, label="有效容量 C_eff / C0（下降）", linewidth=2, color="#4C78A8")
    ax.plot(x, rint, label="等效内阻 R_int / R0（上升）", linewidth=2, color="#F58518")
    ax.set_xlabel("老化进程（等效循环次数/时间，示意）")
    ax.set_ylabel("归一化量")
    ax.set_title("电池老化对预测的结构性影响（示意）：容量衰减 + 内阻上升")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    # annotate impacts
    ax.text(520, 0.86, "容量↓ → 同样掉电量对应更短续航", fontsize=10)
    ax.text(520, 1.45, "内阻↑ → 高负载/低温下电压下垂更强\n→ 可用容量/功率受限更早触发", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aging_effect_concept.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _set_chinese_font()
    plot_portability_matrix()
    plot_aging_concept()
    print("[OK] wrote recommend/extend figures to", str(OUT_DIR))


if __name__ == "__main__":
    main()

