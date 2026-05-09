"""Plot parameter sweep results."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_sweep(input_csv='runs/sweep_results.csv', output_png='runs/sweep_curve.png'):
    """
    Plot parameter sweep results.

    Args:
        input_csv: Path to sweep_results.csv
        output_png: Path to output plot
    """
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"❌ Sweep results file not found: {input_csv}")
        print("   Run parameter sweep first: python tools/param_sweep.py --preset tiny")
        return

    # Read CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Failed to read {input_csv}: {e}")
        return

    # Check if empty
    if len(df) == 0:
        print(f"❌ Sweep results file is empty: {input_csv}")
        return

    # Check required columns
    if 'candidate_id' not in df.columns or 'mean_return' not in df.columns:
        print(f"❌ Sweep results missing required columns")
        return

    # Sort by mean_return (descending) for better visualization
    df_sorted = df.sort_values('mean_return', ascending=False).reset_index(drop=True)

    # Create plot
    plt.figure(figsize=(14, 8))

    # Plot bars
    bars = plt.bar(range(len(df_sorted)), df_sorted['mean_return'],
                    color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)

    # Color bars by performance
    for i, bar in enumerate(bars):
        return_val = df_sorted.iloc[i]['mean_return']
        if return_val >= 100:
            bar.set_color('green')
            bar.set_alpha(0.7)
        elif return_val >= 0:
            bar.set_color('lightgreen')
            bar.set_alpha(0.7)
        elif return_val >= -50:
            bar.set_color('orange')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)

    # Customize plot
    plt.xlabel('Candidate (sorted by performance)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Return', fontsize=12, fontweight='bold')
    plt.title('HalfCheetah CPG Parameter Sweep Results', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df_sorted)), df_sorted['candidate_id'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')

    # Add zero line
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    # Add performance zones
    plt.axhspan(-1000, -100, alpha=0.1, color='red', label='High Cost')
    plt.axhspan(-100, 0, alpha=0.1, color='orange', label='Negative Return')
    plt.axhspan(0, 100, alpha=0.1, color='yellow', label='Low Positive')
    plt.axhspan(100, 1000, alpha=0.1, color='green', label='Working')

    # Legend
    plt.legend(loc='upper right', fontsize=9)

    # Tight layout
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Save plot
    plt.savefig(output_png, dpi=120, bbox_inches='tight')
    print(f"✅ Sweep plot saved to: {output_png}")

    # v0.3新增：如果有mean_x_displacement字段，生成第二张图
    if 'mean_x_displacement' in df.columns and not df['mean_x_displacement'].isna().all():
        plot_displacement_curve(df, input_csv.replace('sweep_results.csv', 'sweep_displacement_curve.png'))

    # Print summary statistics
    print(f"\n📊 Sweep Summary:")
    print(f"   Total candidates: {len(df_sorted)}")
    print(f"   Best return: {df_sorted['mean_return'].max():.2f} ({df_sorted.iloc[0]['candidate_id']})")
    print(f"   Worst return: {df_sorted['mean_return'].min():.2f}")
    print(f"   Positive returns: {(df_sorted['mean_return'] >= 0).sum()}")
    print(f"   Working candidates (>=100): {(df_sorted['mean_return'] >= 100).sum()}")


def plot_displacement_curve(df, output_png='runs/sweep_displacement_curve.png'):
    """
    Plot x displacement curve (v0.3新增).

    Args:
        df: DataFrame with sweep results
        output_png: Path to output plot
    """
    # Filter out rows without displacement data
    df_valid = df.dropna(subset=['mean_x_displacement'])

    if len(df_valid) == 0:
        return  # 没有有效数据，跳过

    # Sort by displacement (descending)
    df_sorted = df_valid.sort_values('mean_x_displacement', ascending=False).reset_index(drop=True)

    # Create plot
    plt.figure(figsize=(14, 8))

    # Plot bars
    bars = plt.bar(range(len(df_sorted)), df_sorted['mean_x_displacement'],
                    color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)

    # Color bars by displacement
    for i, bar in enumerate(bars):
        disp_val = df_sorted.iloc[i]['mean_x_displacement']
        if disp_val >= 1.0:
            bar.set_color('green')
            bar.set_alpha(0.7)
        elif disp_val >= 0.1:
            bar.set_color('lightgreen')
            bar.set_alpha(0.7)
        elif disp_val >= -0.5:
            bar.set_color('orange')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)

    # Customize plot
    plt.xlabel('Candidate (sorted by x displacement)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean X Displacement', fontsize=12, fontweight='bold')
    plt.title('HalfCheetah CPG Parameter Sweep - X Displacement Results', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df_sorted)), df_sorted['candidate_id'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')

    # Add zero line
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    # Add displacement zones
    plt.axhspan(-10, -0.5, alpha=0.1, color='red', label='Moving Backward')
    plt.axhspan(-0.5, 0.1, alpha=0.1, color='orange', label='No Progress')
    plt.axhspan(0.1, 1.0, alpha=0.1, color='yellow', label='Small Progress')
    plt.axhspan(1.0, 10, alpha=0.1, color='green', label='Good Progress')

    # Legend
    plt.legend(loc='upper right', fontsize=9)

    # Tight layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_png, dpi=120, bbox_inches='tight')
    print(f"✅ Displacement plot saved to: {output_png}")

    # Print displacement summary
    print(f"\n🏃 Displacement Summary:")
    print(f"   Valid data: {len(df_sorted)}/{len(df)} candidates")
    print(f"   Best displacement: {df_sorted['mean_x_displacement'].max():.2f} ({df_sorted.iloc[0]['candidate_id']})")
    print(f"   Forward progress: {(df_sorted['mean_x_displacement'] > 0.1).sum()}")
    print(f"   Moving backward: {(df_sorted['mean_x_displacement'] < -0.5).sum()}")


if __name__ == '__main__':
    plot_sweep()
