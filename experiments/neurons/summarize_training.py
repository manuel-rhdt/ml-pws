#!/usr/bin/env python
"""
Summarize training results from neuron model training logs.

Extracts final correlation and validation loss for each trained neuron model.
"""

import pandas as pd
from pathlib import Path
import argparse


def extract_final_metrics(metrics_path: Path) -> dict | None:
    """Extract final correlation and validation loss from a metrics.csv file."""
    try:
        df = pd.read_csv(metrics_path)

        # Get final correlation (last non-null value)
        final_corr = df['corr'].dropna().iloc[-1] if not df['corr'].dropna().empty else None

        # Get final validation loss (last non-null value)
        final_val_loss = df['val_loss'].dropna().iloc[-1] if not df['val_loss'].dropna().empty else None

        # Get final epoch
        final_epoch = df['epoch'].dropna().iloc[-1] if not df['epoch'].dropna().empty else None

        return {
            'final_corr': final_corr,
            'final_val_loss': final_val_loss,
            'final_epoch': int(final_epoch) if final_epoch is not None else None
        }
    except Exception as e:
        print(f"Error reading {metrics_path}: {e}")
        return None


def summarize_training_logs(logs_dir: Path) -> pd.DataFrame:
    """Summarize training logs from all neuron directories."""
    results = []

    # Find all neuron directories
    neuron_dirs = sorted(logs_dir.glob("neuron_*"), key=lambda x: int(x.name.split('_')[1]))

    for neuron_dir in neuron_dirs:
        metrics_path = neuron_dir / "metrics.csv"

        if not metrics_path.exists():
            print(f"Warning: No metrics.csv found in {neuron_dir}")
            continue

        neuron_id = int(neuron_dir.name.split('_')[1])
        metrics = extract_final_metrics(metrics_path)

        if metrics:
            results.append({
                'neuron_id': neuron_id,
                'final_corr': metrics['final_corr'],
                'final_val_loss': metrics['final_val_loss'],
                'final_epoch': metrics['final_epoch']
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Summarize neuron model training results')
    parser.add_argument('--logs-dir', type=Path,
                        default=Path(__file__).parent / 'training_logs',
                        help='Directory containing training logs')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output CSV file path (optional)')
    parser.add_argument('--sort-by', choices=['neuron_id', 'final_corr', 'final_val_loss'],
                        default='neuron_id', help='Column to sort results by')
    parser.add_argument('--ascending', action='store_true', default=True,
                        help='Sort in ascending order')
    args = parser.parse_args()

    # Summarize training logs
    df = summarize_training_logs(args.logs_dir)

    if df.empty:
        print("No training logs found!")
        return

    # Sort results
    ascending = args.ascending if args.sort_by == 'neuron_id' else not args.ascending
    if args.sort_by == 'final_corr':
        ascending = False  # Higher correlation is better
    elif args.sort_by == 'final_val_loss':
        ascending = True   # Lower loss is better
    df = df.sort_values(args.sort_by, ascending=ascending)

    # Print summary statistics
    print("=" * 70)
    print("NEURON MODEL TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nTotal models trained: {len(df)}")
    print(f"\nCorrelation Statistics:")
    print(f"  Mean:   {df['final_corr'].mean():.4f}")
    print(f"  Std:    {df['final_corr'].std():.4f}")
    print(f"  Min:    {df['final_corr'].min():.4f}")
    print(f"  Max:    {df['final_corr'].max():.4f}")
    print(f"  Median: {df['final_corr'].median():.4f}")

    print(f"\nValidation Loss Statistics:")
    print(f"  Mean:   {df['final_val_loss'].mean():.6f}")
    print(f"  Std:    {df['final_val_loss'].std():.6f}")
    print(f"  Min:    {df['final_val_loss'].min():.6f}")
    print(f"  Max:    {df['final_val_loss'].max():.6f}")
    print(f"  Median: {df['final_val_loss'].median():.6f}")

    # Print top and bottom performers
    print("\n" + "=" * 70)
    print("TOP 10 MODELS (by correlation)")
    print("=" * 70)
    top_10 = df.nlargest(10, 'final_corr')
    print(top_10.to_string(index=False))

    print("\n" + "=" * 70)
    print("BOTTOM 10 MODELS (by correlation)")
    print("=" * 70)
    bottom_10 = df.nsmallest(10, 'final_corr')
    print(bottom_10.to_string(index=False))

    # Print full table
    print("\n" + "=" * 70)
    print("ALL MODELS")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save to CSV if output path provided
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
