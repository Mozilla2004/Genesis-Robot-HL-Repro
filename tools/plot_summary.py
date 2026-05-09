"""Plot summary curve from runs/summary.csv."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_summary(input_csv='runs/summary.csv', output_png='runs/summary_curve.png'):
    """
    Plot mean_return vs trial order from summary CSV.

    Args:
        input_csv: Path to summary.csv
        output_png: Path to output plot
    """
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"❌ Summary file not found: {input_csv}")
        print("   Run some trials first: python run_eval.py --policy cpg_pd --episodes 3")
        return

    # Read CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Failed to read {input_csv}: {e}")
        return

    # Check if empty
    if len(df) == 0:
        print(f"❌ Summary file is empty: {input_csv}")
        return

    # Check required columns
    if 'mean_return' not in df.columns:
        print(f"❌ Summary missing 'mean_return' column")
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['mean_return'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Trial Order')
    plt.ylabel('Mean Return')
    plt.title('HalfCheetah Performance Over Trials')
    plt.grid(True, alpha=0.3)

    # Add policy legend if available
    if 'policy' in df.columns:
        policies = df['policy'].unique()
        plt.legend(policies)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_png, dpi=100)
    print(f"✅ Plot saved to: {output_png}")


if __name__ == '__main__':
    plot_summary()
