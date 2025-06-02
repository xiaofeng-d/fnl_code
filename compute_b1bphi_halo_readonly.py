#!/usr/bin/env python3
"""
Plot b1 vs bphi comparison using previously calculated data.
This script loads .npz files and creates high-quality visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
import argparse
from pathlib import Path

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
# mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['figure.dpi'] = 300

def find_latest_npz(pattern="bias_ml*_mh*_snap*.npz"):
    """Find the latest npz file"""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found")
    
    # Sort by modification time, return the latest
    return max(files, key=os.path.getmtime)

def format_mass(m):
    """Format mass value, removing plus sign from scientific notation"""
    return f"{m:.2e}".replace('+', '')

def extract_mass_from_filename(filename):
    """Extract mass range from filename"""
    import re
    match = re.search(r'ml(\d+\.\d+e[+-]?\d+)_mh(\d+\.\d+e[+-]?\d+)', filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def plot_b1_vs_bphi(files=None, output_file=None, title=None, 
                   figsize=(10, 8), dpi=300, add_theory_line=True,
                   xlim=None, ylim=None, annotate=True):
    """
    Plot b1 vs bphi comparison
    
    Parameters:
        files (list): List of data file paths, if None will find all matching files
        output_file (str): Output image file path
        title (str): Plot title
        figsize (tuple): Figure size
        dpi (int): Figure resolution
        add_theory_line (bool): Whether to add theoretical prediction line
        xlim, ylim (tuple): Axis limits
        annotate (bool): Whether to annotate mass values
    """
    # If no files specified, find all matching files
    if files is None:
        files = sorted(glob.glob("bias_ml*_mh*_snap*.npz"))
        if not files:
            raise FileNotFoundError("No matching data files found")
    elif isinstance(files, str):
        files = [files]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Store all data points for later analysis
    all_b1 = []
    all_bphi = []
    all_masses = []
    
    # Process each file
    for file in files:
        print(f"Processing file: {file}")
        data = np.load(file)
        
        # Extract b1 and bphi values
        b1_mean = data['b1_mean']
        bphi_mean = data['bphi_mean']
        
        # Extract mass range
        ml, mh = extract_mass_from_filename(file)
        mass_avg = np.sqrt(ml * mh)  # geometric mean
        
        # Store data
        all_b1.append(b1_mean)
        all_bphi.append(bphi_mean)
        all_masses.append(mass_avg)
        
        # Plot data point
        label = f"M = {format_mass(mass_avg)} M⊙/h"
        ax.scatter(b1_mean, bphi_mean, s=100, label=label)
        
        # Annotate mass value
        if annotate:
            ax.annotate(f"{format_mass(mass_avg)}", 
                       (b1_mean, bphi_mean),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9)
    
    # Convert to arrays
    all_b1 = np.array(all_b1)
    all_bphi = np.array(all_bphi)
    all_masses = np.array(all_masses)
    
    # Add theoretical prediction line (bphi = 2(b1-1))
    if add_theory_line:
        # Calculate appropriate x range
        if xlim:
            x_range = np.linspace(xlim[0], xlim[1], 100)
        else:
            x_min = max(0.5, min(all_b1) * 0.8)
            x_max = max(all_b1) * 1.2
            x_range = np.linspace(x_min, x_max, 100)
        
        # Plot theory line
        y_theory = 2 * (x_range - 1)
        ax.plot(x_range, y_theory, 'k--', lw=2, label="Theory: $b_φ = 2*\delta_c(b_1-1)$")
    
    # Set axes
    ax.set_xlabel(r"Linear bias $b_1$", fontsize=16)
    ax.set_ylabel(r"Growth bias $b_φ$", fontsize=16)
    
    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Add grid and legend
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=12, loc='best')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=18)
    else:
        ax.set_title("Linear bias vs Growth bias", fontsize=18)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='-', alpha=0.3)
    
    # Save figure
    if output_file is None:
        output_file = "b1_vs_bphi_comparison.png"
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Number of data points: {len(all_b1)}")
    print(f"b1 range: {min(all_b1):.3f} to {max(all_b1):.3f}")
    print(f"bphi range: {min(all_bphi):.3f} to {max(all_bphi):.3f}")
    
    # Calculate deviation from theoretical prediction
    theory_bphi = 2 * 1.686 * (all_b1 - 1)
    deviation = all_bphi - theory_bphi
    mean_deviation = np.mean(deviation)
    std_deviation = np.std(deviation)
    print(f"Mean deviation from theory: {mean_deviation:.3f} ± {std_deviation:.3f}")
    
    return fig, ax, (all_masses, all_b1, all_bphi)

def main():
    parser = argparse.ArgumentParser(description="Plot b1 vs bphi comparison")
    parser.add_argument('--files', '-f', nargs='+', help='List of data file paths')
    parser.add_argument('--output', '-o', default="b1_vs_bphi_comparison.png", help='Output image file path')
    parser.add_argument('--title', '-t', help='Plot title')
    parser.add_argument('--width', type=float, default=10, help='Figure width')
    parser.add_argument('--height', type=float, default=8, help='Figure height')
    parser.add_argument('--dpi', type=int, default=300, help='Figure resolution')
    parser.add_argument('--no-theory', action='store_false', dest='theory', help='Do not show theory line')
    parser.add_argument('--no-annotate', action='store_false', dest='annotate', help='Do not annotate mass values')
    parser.add_argument('--xmin', type=float, help='x-axis minimum')
    parser.add_argument('--xmax', type=float, help='x-axis maximum')
    parser.add_argument('--ymin', type=float, help='y-axis minimum')
    parser.add_argument('--ymax', type=float, help='y-axis maximum')
    args = parser.parse_args()
    
    # Set axis limits
    xlim = (-1, 30) #None if args.xmin is None or args.xmax is None else (args.xmin, args.xmax)
    ylim = (-1, 20) #None if args.ymin is None or args.ymax is None else (args.ymin, args.ymax)
    
    # Plot figure
    plot_b1_vs_bphi(
        files=args.files,
        output_file=args.output,
        title=args.title,
        figsize=(args.width, args.height),
        dpi=args.dpi,
        add_theory_line=args.theory,
        xlim=xlim,
        ylim=ylim,
        annotate=args.annotate
    )

if __name__ == "__main__":
    main()
