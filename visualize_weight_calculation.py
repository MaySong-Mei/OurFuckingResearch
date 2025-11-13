"""
可视化一致性权重的计算过程
展示：方差 → sigmoid映射 → 最终权重
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# 设置风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_weight_computation():
    """可视化权重计算的全过程"""

    # ============ 参数设定 ============
    w_min = 0.5
    w_max = 2.0
    tau = 0.15
    kappa = 0.4

    # ============ 1. 方差 → 权重映射 ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Consistency Weight Calculation Process', fontsize=16, fontweight='bold')

    # Subplot 1: 方差 vs 权重映射
    u_values = np.linspace(0, 0.4, 200)
    sigmoid_inputs = (u_values - tau) / kappa
    sigmoid_outputs = 1 / (1 + np.exp(-sigmoid_inputs))
    weights = w_min + (w_max - w_min) * sigmoid_outputs

    ax = axes[0, 0]
    ax.plot(u_values, weights, 'b-', linewidth=2.5, label='Weight function')
    ax.axvline(tau, color='red', linestyle='--', linewidth=2, label=f'τ (threshold) = {tau}')
    ax.axhline(w_min, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'w_min = {w_min}')
    ax.axhline(w_max, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'w_max = {w_max}')
    ax.fill_between(u_values, w_min, weights, alpha=0.2, color='blue')
    ax.set_xlabel('Inconsistency metric u(i,j)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Weight w(i,j)', fontsize=11, fontweight='bold')
    ax.set_title('Weight Mapping Function', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.4, 2.1)

    # Subplot 2: Sigmoid函数 (归一化)
    ax = axes[0, 1]
    sigmoid_values = sigmoid_outputs
    ax.plot(u_values, sigmoid_values, 'g-', linewidth=2.5, label='sigmoid()')
    ax.axvline(tau, color='red', linestyle='--', linewidth=2, label=f'τ = {tau}')
    ax.axhline(0.5, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.fill_between(u_values, 0, sigmoid_values, alpha=0.2, color='green')
    ax.set_xlabel('Inconsistency metric u(i,j)', fontsize=11, fontweight='bold')
    ax.set_ylabel('sigmoid output (0~1)', fontsize=11, fontweight='bold')
    ax.set_title('Sigmoid Normalization', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1.05)

    # Subplot 3: 权重分布直方图 (模拟)
    ax = axes[1, 0]
    # 模拟不同场景的方差分布
    consistent_u = np.random.normal(loc=0.05, scale=0.02, size=3000)
    uncertain_u = np.random.normal(loc=0.15, scale=0.04, size=3000)
    inconsistent_u = np.random.normal(loc=0.30, scale=0.05, size=3000)

    consistent_w = w_min + (w_max - w_min) * (1 / (1 + np.exp(-(consistent_u - tau) / kappa)))
    uncertain_w = w_min + (w_max - w_min) * (1 / (1 + np.exp(-(uncertain_u - tau) / kappa)))
    inconsistent_w = w_min + (w_max - w_min) * (1 / (1 + np.exp(-(inconsistent_u - tau) / kappa)))

    ax.hist(consistent_w, bins=30, alpha=0.6, label='Consistent regions', color='green', edgecolor='black')
    ax.hist(uncertain_w, bins=30, alpha=0.6, label='Uncertain regions', color='yellow', edgecolor='black')
    ax.hist(inconsistent_w, bins=30, alpha=0.6, label='Inconsistent regions', color='red', edgecolor='black')
    ax.set_xlabel('Weight value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pixel count', fontsize=11, fontweight='bold')
    ax.set_title('Weight Distribution by Region Type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 4: 权重范围和说明
    ax = axes[1, 1]
    ax.axis('off')

    # 创建文本说明
    info_text = f"""
WEIGHT CALCULATION FORMULA
═══════════════════════════════════

Step 1: Stack 3-view probabilities
   Input: prob_axial, prob_sagittal, prob_coronal
   Shape: [B, N, H, W, C, 3]

Step 2: Compute class variance
   var(c) = var(prob_axial[c], prob_sagittal[c], prob_coronal[c])
   Shape: [B, N, H, W, C]

Step 3: Sum to inconsistency metric
   u(i,j) = Σ_c var(c)
   Shape: [B, N, H, W]

Step 4: Map via sigmoid
   w(i,j) = w_min + (w_max - w_min) × sigmoid((u - τ) / κ)

   where:
   • w_min = {w_min}   (consistent regions)
   • w_max = {w_max}   (inconsistent regions)
   • τ = {tau}         (variance tolerance)
   • κ = {kappa}       (sigmoid steepness)

WEIGHT INTERPRETATION
══════════════════════

• u < 0.05  → w ≈ 0.5-0.7   (trust, low supervision)
• u ≈ 0.15  → w ≈ 1.2-1.3   (uncertain, medium)
• u > 0.25  → w ≈ 1.5-2.0   (distrust, high supervision)

APPLICATION IN LOSS
════════════════════

L_interp = mean(w * |I_interp - I_gt|)

Weighted loss = unweighted_loss × weight
"""

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = Path('/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/OurFuckingResearch/weight_calculation_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def visualize_numerical_example():
    """可视化具体数值例子"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Numerical Example: Consistency Weight Calculation', fontsize=14, fontweight='bold')

    w_min, w_max, tau, kappa = 0.5, 2.0, 0.15, 0.4

    # ============ 场景1：一致 ============
    ax = axes[0]

    scenarios = {
        'Scenario 1:\nConsistent': {
            'probs': [[0.90, 0.88, 0.92], [0.10, 0.12, 0.08]],
            'color': 'green',
            'pos': 0
        },
        'Scenario 2:\nUncertain': {
            'probs': [[0.80, 0.75, 0.70], [0.20, 0.25, 0.30]],
            'color': 'yellow',
            'pos': 1
        },
        'Scenario 3:\nInconsistent': {
            'probs': [[0.95, 0.20, 0.15], [0.05, 0.80, 0.85]],
            'color': 'red',
            'pos': 2
        }
    }

    # 为三个场景计算权重
    results = []
    for scenario_name, scenario_info in scenarios.items():
        probs_array = np.array(scenario_info['probs'])

        # 计算每个类的方差
        var_foreground = np.var(probs_array[0])
        var_background = np.var(probs_array[1])

        # 总方差
        u = var_foreground + var_background

        # 计算权重
        sigmoid_input = (u - tau) / kappa
        sigmoid_output = 1 / (1 + np.exp(-sigmoid_input))
        weight = w_min + (w_max - w_min) * sigmoid_output

        results.append({
            'name': scenario_name.split(':')[0].strip(),
            'probs': probs_array,
            'var_fg': var_foreground,
            'var_bg': var_background,
            'u': u,
            'weight': weight,
            'color': scenario_info['color'],
            'pos': scenario_info['pos']
        })

    # 绘制三个场景并排
    for result in results:
        ax = axes[result['pos']]

        # 标题
        title = result['name']
        ax.set_title(title, fontsize=12, fontweight='bold', color=result['color'])

        # 概率条形图
        views = ['Axial', 'Sagittal', 'Coronal']
        x = np.arange(len(views))
        width = 0.35

        ax.bar(x - width/2, result['probs'][0], width, label='Foreground', color='#FF6B6B', alpha=0.8)
        ax.bar(x + width/2, result['probs'][1], width, label='Background', color='#4ECDC4', alpha=0.8)

        ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(views, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加统计信息
        stats_text = f"""var_fg = {result['var_fg']:.5f}
var_bg = {result['var_bg']:.5f}
───────────────
u = {result['u']:.5f}

w = {result['weight']:.3f}"""

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    plt.tight_layout()
    output_path = Path('/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/OurFuckingResearch/weight_example_scenarios.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved example scenarios to {output_path}")
    plt.close()


def visualize_parameter_sensitivity():
    """可视化参数对权重的影响"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')

    u_values = np.linspace(0, 0.4, 200)

    # ============ 1. tau的影响 ============
    ax = axes[0, 0]
    for tau in [0.05, 0.15, 0.25, 0.35]:
        w_min, w_max, kappa = 0.5, 2.0, 0.4
        sigmoid_input = (u_values - tau) / kappa
        weights = w_min + (w_max - w_min) / (1 + np.exp(-sigmoid_input))
        ax.plot(u_values, weights, linewidth=2, label=f'τ={tau}')

    ax.set_xlabel('Inconsistency metric u', fontsize=10, fontweight='bold')
    ax.set_ylabel('Weight w', fontsize=10, fontweight='bold')
    ax.set_title('Effect of τ (Tolerance)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.4, 2.1)

    # ============ 2. kappa的影响 ============
    ax = axes[0, 1]
    for kappa in [0.2, 0.4, 0.6, 0.8]:
        w_min, w_max, tau = 0.5, 2.0, 0.15
        sigmoid_input = (u_values - tau) / kappa
        weights = w_min + (w_max - w_min) / (1 + np.exp(-sigmoid_input))
        ax.plot(u_values, weights, linewidth=2, label=f'κ={kappa}')

    ax.set_xlabel('Inconsistency metric u', fontsize=10, fontweight='bold')
    ax.set_ylabel('Weight w', fontsize=10, fontweight='bold')
    ax.set_title('Effect of κ (Steepness)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.4, 2.1)

    # ============ 3. w_min的影响 ============
    ax = axes[1, 0]
    tau, kappa = 0.15, 0.4
    for w_min in [0.3, 0.5, 0.7, 0.9]:
        w_max = 2.0
        sigmoid_input = (u_values - tau) / kappa
        weights = w_min + (w_max - w_min) / (1 + np.exp(-sigmoid_input))
        ax.plot(u_values, weights, linewidth=2, label=f'w_min={w_min}')

    ax.set_xlabel('Inconsistency metric u', fontsize=10, fontweight='bold')
    ax.set_ylabel('Weight w', fontsize=10, fontweight='bold')
    ax.set_title('Effect of w_min (Lower Bound)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.2, 2.1)

    # ============ 4. w_max的影响 ============
    ax = axes[1, 1]
    w_min, tau, kappa = 0.5, 0.15, 0.4
    for w_max in [1.5, 2.0, 2.5, 3.0]:
        sigmoid_input = (u_values - tau) / kappa
        weights = w_min + (w_max - w_min) / (1 + np.exp(-sigmoid_input))
        ax.plot(u_values, weights, linewidth=2, label=f'w_max={w_max}')

    ax.set_xlabel('Inconsistency metric u', fontsize=10, fontweight='bold')
    ax.set_ylabel('Weight w', fontsize=10, fontweight='bold')
    ax.set_title('Effect of w_max (Upper Bound)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.4, 3.2)

    plt.tight_layout()
    output_path = Path('/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/OurFuckingResearch/parameter_sensitivity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved parameter sensitivity to {output_path}")
    plt.close()


def create_comparison_table():
    """创建权重计算对比表"""

    w_min, w_max, tau, kappa = 0.5, 2.0, 0.15, 0.4

    scenarios = [
        {
            'name': 'Consistent',
            'probs': [[0.90, 0.88, 0.92], [0.10, 0.12, 0.08]],
            'description': '三视图高度一致'
        },
        {
            'name': 'Low-Inconsistent',
            'probs': [[0.85, 0.82, 0.80], [0.15, 0.18, 0.20]],
            'description': '略有差异'
        },
        {
            'name': 'Medium-Inconsistent',
            'probs': [[0.80, 0.70, 0.60], [0.20, 0.30, 0.40]],
            'description': '中等差异'
        },
        {
            'name': 'High-Inconsistent',
            'probs': [[0.95, 0.20, 0.15], [0.05, 0.80, 0.85]],
            'description': '高度不一致'
        }
    ]

    print("\n" + "="*100)
    print("CONSISTENCY WEIGHT CALCULATION TABLE")
    print("="*100)
    print(f"\nParameters: w_min={w_min}, w_max={w_max}, τ={tau}, κ={kappa}\n")

    print(f"{'Scenario':<20} {'Prob Axial':<20} {'Prob Sagit':<20} {'Prob Coro':<20} "
          f"{'var_fg':<10} {'var_bg':<10} {'u':<10} {'w(u)':<10} {'Description':<20}")
    print("-" * 140)

    for scenario in scenarios:
        probs = np.array(scenario['probs'])
        var_fg = np.var(probs[0])
        var_bg = np.var(probs[1])
        u = var_fg + var_bg
        sigmoid_input = (u - tau) / kappa
        weight = w_min + (w_max - w_min) / (1 + np.exp(-sigmoid_input))

        prob_axial = f"[{probs[0,0]:.2f}, {probs[1,0]:.2f}]"
        prob_sagit = f"[{probs[0,1]:.2f}, {probs[1,1]:.2f}]"
        prob_coro = f"[{probs[0,2]:.2f}, {probs[1,2]:.2f}]"

        print(f"{scenario['name']:<20} {prob_axial:<20} {prob_sagit:<20} {prob_coro:<20} "
              f"{var_fg:<10.5f} {var_bg:<10.5f} {u:<10.5f} {weight:<10.3f} {scenario['description']:<20}")

    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    print("开始生成权重计算可视化...")

    # 生成主要的权重函数可视化
    visualize_weight_computation()

    # 生成数值例子可视化
    visualize_numerical_example()

    # 生成参数敏感性分析
    visualize_parameter_sensitivity()

    # 创建对比表
    create_comparison_table()

    print("\n✓ 所有可视化已完成！")
    print("  生成的文件：")
    print("  • weight_calculation_visualization.png")
    print("  • weight_example_scenarios.png")
    print("  • parameter_sensitivity.png")
