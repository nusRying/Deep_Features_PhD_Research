import argparse
import pandas as pd
import os
import sys
from .plotting import (
    plot_performance, 
    plot_population, 
    plot_sets, 
    plot_operations, 
    plot_timing
)

def main():
    parser = argparse.ArgumentParser(description='ExSTraCS Training Visualization Tool')
    parser.add_argument('csv_path', type=str, help='Path to the iterationData.csv file')
    parser.add_argument('--out', type=str, default='plots', help='Directory to save plots (default: plots)')
    parser.add_argument('--mavg', type=int, default=300, help='Moving average threshold for set sizes (default: 300)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found at {args.csv_path}")
        sys.exit(1)
        
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        print(f"Created output directory: {args.out}")
        
    print(f"Loading data from {args.csv_path}...")
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    print("Generating performance plots...")
    plot_performance(df, save_path=args.out)
    
    print("Generating population plots...")
    plot_population(df, save_path=args.out)
    
    print("Generating set size plots...")
    plot_sets(df, threshold=args.mavg, save_path=args.out)
    
    print("Generating operations plots...")
    plot_operations(df, save_path=args.out)
    
    print("Generating timing plots...")
    plot_timing(df, save_path=args.out)
    
    print(f"Done! All plots saved to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
