"""
Visualization script for the variance-based consistency weighting function

This script generates plots showing:
1. Sigmoid mapping curve with different parameters
2. Weight distribution for different scenarios
3. Parameter sensitivity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from losses import SegmentationConsistencyWeighting


def sigmoid(x):
    """Numpy implementation of sigmoid"""
    return 1 / (1 + np.exp(-x))


def plot_sigmoid_mapping(w_min=0.5, w_max=2.0, tau=0.15, kappa=0.4):
    """Plot the sigmoid mapping function"""
    variance_range = np.linspace(0, 0.5, 1000)
    weights = w_min + (w_max - w_min) * sigmoid((variance_range - tau) / kappa)

    plt.figure(figsize=(10, 6))
    plt.plot(variance_range, weights, 'b-', linewidth=2.5, label='Weight function')
    plt.axhline(y=w_min, color='g', linestyle='--', alpha=0.7, label=f'w_min = {w_min}')
    plt.axhline(y=w_max, color='r', linestyle='--', alpha=0.7, label=f'w_max = {w_max}')
    plt.axvline(x=tau, color='orange', linestyle='--', alpha=0.7, label=f'tau = {tau} (tolerance)')

    # Mark special points
    y_at_tau = w_min + (w_max - w_min) * sigmoid(0)
    plt.plot(tau, y_at_tau, 'ro', markersize=10, label=f'Weight at tau: {y_at_tau:.3f}')

    plt.xlabel('Inconsistency Metric u(i) (variance)', fontsize=12)
    plt.ylabel('Weight w(i)', fontsize=12)
    plt.title(f'Variance-Sigmoid Weight Mapping\n(w_min={w_min}, w_max={w_max}, tau={tau}, kappa={kappa})',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('weight_sigmoid_mapping.png', dpi=150)
    print("✓ Saved: weight_sigmoid_mapping.png")
    plt.close()


def plot_parameter_sensitivity():
    """Plot sensitivity to different parameters"""
    variance_range = np.linspace(0, 0.5, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

    # w_min sensitivity
    ax = axes[0, 0]
    for w_min in [0.3, 0.5, 0.7]:
        weights = w_min + (2.0 - w_min) * sigmoid((variance_range - 0.15) / 0.4)
        ax.plot(variance_range, weights, label=f'w_min={w_min}', linewidth=2)
    ax.set_xlabel('Variance u(i)', fontsize=11)
    ax.set_ylabel('Weight w(i)', fontsize=11)
    ax.set_title('Effect of w_min (lower limit)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # w_max sensitivity
    ax = axes[0, 1]
    for w_max in [1.5, 2.0, 3.0]:
        weights = 0.5 + (w_max - 0.5) * sigmoid((variance_range - 0.15) / 0.4)
        ax.plot(variance_range, weights, label=f'w_max={w_max}', linewidth=2)
    ax.set_xlabel('Variance u(i)', fontsize=11)
    ax.set_ylabel('Weight w(i)', fontsize=11)
    ax.set_title('Effect of w_max (upper limit)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # tau sensitivity (tolerance threshold)
    ax = axes[1, 0]
    for tau in [0.05, 0.15, 0.25]:
        weights = 0.5 + (2.0 - 0.5) * sigmoid((variance_range - tau) / 0.4)
        ax.plot(variance_range, weights, label=f'tau={tau}', linewidth=2)
    ax.axvline(x=0.05, color='C0', linestyle=':', alpha=0.5)
    ax.axvline(x=0.15, color='C1', linestyle=':', alpha=0.5)
    ax.axvline(x=0.25, color='C2', linestyle=':', alpha=0.5)
    ax.set_xlabel('Variance u(i)', fontsize=11)
    ax.set_ylabel('Weight w(i)', fontsize=11)
    ax.set_title('Effect of tau (tolerance threshold)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # kappa sensitivity (smoothness)
    ax = axes[1, 1]
    for kappa in [0.2, 0.4, 0.6]:
        weights = 0.5 + (2.0 - 0.5) * sigmoid((variance_range - 0.15) / kappa)
        ax.plot(variance_range, weights, label=f'kappa={kappa}', linewidth=2)
    ax.set_xlabel('Variance u(i)', fontsize=11)
    ax.set_ylabel('Weight w(i)', fontsize=11)
    ax.set_title('Effect of kappa (transition sharpness)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=150)
    print("✓ Saved: parameter_sensitivity.png")
    plt.close()


def plot_preset_configurations():
    """Compare preset configurations"""
    variance_range = np.linspace(0, 0.5, 1000)

    configs = {
        'Weak (Loose)': {'w_min': 0.7, 'w_max': 1.5, 'tau': 0.20, 'kappa': 0.5},
        'Standard (Recommended)': {'w_min': 0.5, 'w_max': 2.0, 'tau': 0.15, 'kappa': 0.4},
        'Strong (Strict)': {'w_min': 0.4, 'w_max': 3.0, 'tau': 0.10, 'kappa': 0.3},
    }

    plt.figure(figsize=(12, 6))
    colors = ['green', 'blue', 'red']

    for (name, params), color in zip(configs.items(), colors):
        weights = params['w_min'] + (params['w_max'] - params['w_min']) * \
                  sigmoid((variance_range - params['tau']) / params['kappa'])
        plt.plot(variance_range, weights, linewidth=2.5, color=color, label=name)

    plt.xlabel('Inconsistency Metric u(i) (variance)', fontsize=12)
    plt.ylabel('Weight w(i)', fontsize=12)
    plt.title('Preset Weight Configurations Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('preset_configurations.png', dpi=150)
    print("✓ Saved: preset_configurations.png")
    plt.close()


def plot_weight_distribution(batch_size=2, num_slices=3, height=8, width=8):
    """Simulate and plot weight distribution for different consistency scenarios"""
    weighter = SegmentationConsistencyWeighting(w_min=0.5, w_max=2.0, tau=0.15, kappa=0.4)

    scenarios = {}

    # Scenario 1: Perfectly consistent
    prob = torch.ones(batch_size, num_slices, height, width, 2)
    prob[..., 0] = 0.3
    prob[..., 1] = 0.7
    weights1 = weighter(prob, prob, prob)
    scenarios['Consistent'] = weights1.detach().cpu().numpy().flatten()

    # Scenario 2: Highly inconsistent
    prob1 = torch.zeros(batch_size, num_slices, height, width, 2)
    prob1[..., 1] = 0.9
    prob2 = torch.zeros(batch_size, num_slices, height, width, 2)
    prob2[..., 0] = 0.9
    prob3 = torch.ones(batch_size, num_slices, height, width, 2) * 0.5
    weights2 = weighter(prob1, prob2, prob3)
    scenarios['Inconsistent'] = weights2.detach().cpu().numpy().flatten()

    # Scenario 3: Random variance
    probs = [F.softmax(torch.randn(batch_size, num_slices, height, width, 2) * 0.5, dim=-1)
             for _ in range(3)]
    weights3 = weighter(probs[0], probs[1], probs[2])
    scenarios['Mixed (Random)'] = weights3.detach().cpu().numpy().flatten()

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Weight Distribution for Different Consistency Scenarios', fontsize=14, fontweight='bold')

    colors = ['green', 'red', 'blue']
    for ax, (name, weights), color in zip(axes, scenarios.items(), colors):
        ax.hist(weights, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(weights.mean(), color='darkred', linestyle='--', linewidth=2, label=f'mean={weights.mean():.3f}')
        ax.set_xlabel('Weight Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('weight_distribution.png', dpi=150)
    print("✓ Saved: weight_distribution.png")
    plt.close()


def print_summary():
    """Print summary of visualizations"""
    print("\n" + "=" * 70)
    print("WEIGHT FUNCTION VISUALIZATION SUMMARY")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  1. weight_sigmoid_mapping.png")
    print("     └─ Shows the sigmoid mapping curve with default parameters")
    print("\n  2. parameter_sensitivity.png")
    print("     └─ 2x2 grid showing sensitivity to each parameter")
    print("     └─ w_min: lower weight limit")
    print("     └─ w_max: upper weight limit")
    print("     └─ tau: tolerance threshold")
    print("     └─ kappa: transition smoothness")
    print("\n  3. preset_configurations.png")
    print("     └─ Comparison of weak, standard, and strong configurations")
    print("\n  4. weight_distribution.png")
    print("     └─ Histograms of weights for different consistency scenarios")
    print("     └─ Consistent: all views agree")
    print("     └─ Inconsistent: views highly disagree")
    print("     └─ Mixed: random variance")
    print("\n" + "=" * 70)
    print("Interpretation Guide:")
    print("=" * 70)
    print("✓ Steeper curves = stronger response to inconsistency")
    print("✓ Higher w_min = more weight on consistent regions")
    print("✓ Higher w_max = more punishment for inconsistent regions")
    print("✓ Larger tau = more tolerance for small differences")
    print("✓ Smaller kappa = sharper transition between w_min and w_max")
    print("=" * 70 + "\n")


def main():
    """Generate all visualizations"""
    print("Generating weight function visualizations...")
    print()

    plot_sigmoid_mapping()
    plot_parameter_sensitivity()
    plot_preset_configurations()
    plot_weight_distribution()

    print_summary()


if __name__ == '__main__':
    main()
