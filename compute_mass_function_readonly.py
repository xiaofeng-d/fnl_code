#!/usr/bin/env python3
"""
Plot halo mass function from previously saved data.
This script reads .npz files containing mass function data and creates visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
from pathlib import Path
import sys

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['figure.dpi'] = 300

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION - EDIT THESE VALUES DIRECTLY
# ─────────────────────────────────────────────────────────────────────────────
# Snapshot to plot
SNAP = 624  # ROCKSTAR step number (624=z0, 310=z1, 205=z2)

# Input data directory (where .npz files are stored)
DATA_DIR = "/home/ac.xdong/hacc-bispec/fnl-paper-plots/mass_function_data"

# Simulation parameters (for labels)
SIGMA8_G = 0.834   # baseline σ₈
SIGMA8_H = 0.844   # high‑σ₈ value
SIGMA8_L = 0.824   # low‑σ₈ value

# Output settings
PLOT_TITLE = None                  # Custom plot title (None = auto-generate)
FIGURE_WIDTH = 10                  # Figure width in inches
FIGURE_HEIGHT = 8                  # Figure height in inches
FIGURE_DPI = 300                   # Figure resolution
ADD_THEORY = False                 # Whether to add theoretical predictions

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def format_mass(m):
    """Format mass value, removing plus sign from scientific notation"""
    return f"{m:.2e}".replace('+', '')

def find_mass_function_files(data_dir, snap=None, sim_name=None):
    """
    Find mass function data files in the specified directory
    
    Parameters:
        data_dir: Directory containing .npz files
        snap: Optional snapshot number to filter by
        sim_name: Optional simulation name to filter by
        
    Returns:
        List of file paths
    """
    pattern = "mass_function_"
    if sim_name:
        pattern += f"{sim_name}_"
    pattern += "*"
    if snap:
        pattern += f"snap{snap}.npz"
    else:
        pattern += ".npz"
    
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    
    if not files:
        print(f"Warning: No mass function data files found matching {pattern}")
    else:
        print(f"Found {len(files)} mass function data files")
    
    return files

def load_mass_function_data(file_path):
    """
    Load mass function data from a .npz file
    
    Parameters:
        file_path: Path to the .npz file
        
    Returns:
        Dictionary with mass function data
    """
    try:
        data = np.load(file_path)
        result = {key: data[key] for key in data.files}
        
        # Extract simulation name and snapshot from filename if not in data
        if 'sim_name' not in result:
            import re
            match = re.search(r'mass_function_([^_]+)_.*snap(\d+)', os.path.basename(file_path))
            if match:
                result['sim_name'] = match.group(1)
                result['snap'] = int(match.group(2))
        
        return result
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_sim_label(sim_name):
    """Get a descriptive label for a simulation"""
    if sim_name == 'gauss':
        return f"Gaussian (σ₈={SIGMA8_G})"
    elif sim_name == 's8h':
        return f"High σ₈ ({SIGMA8_H})"
    elif sim_name == 's8l':
        return f"Low σ₈ ({SIGMA8_L})"
    else:
        return sim_name.capitalize()

def plot_mass_functions(data_files, output_file=None, title=None, 
                       figsize=(10, 8), dpi=300, add_theory=False):
    """
    Plot mass functions from data files
    
    Parameters:
        data_files: List of data file paths
        output_file: Output image file path
        title: Plot title
        figsize: Figure size
        dpi: Figure resolution
        add_theory: Whether to add theoretical mass function predictions
    """
    # Load data from files
    all_data = []
    for file in data_files:
        data = load_mass_function_data(file)
        if data:
            all_data.append(data)
    
    if not all_data:
        print("No valid data files found")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get snapshot and z value from first file
    snap = all_data[0].get('snap', SNAP)
    z_value = all_data[0].get('z', 0)
    
    # Store data for ratio plot
    all_bin_centers = []
    all_mass_functions = []
    all_labels = []
    
    # Plot each mass function
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, data in enumerate(all_data):
        # Extract data
        bin_centers = data['bin_centers']
        mass_function = data['mass_function']
        number_counts = data['number_counts']
        sim_name = data.get('sim_name', f"sim{i}")
        
        # Store data for ratio plot
        all_bin_centers.append(bin_centers)
        all_mass_functions.append(mass_function)
        
        # Get label
        label = get_sim_label(sim_name)
        all_labels.append(label)
        
        # Plot mass function
        color = colors[i % len(colors)]
        ax.loglog(bin_centers, mass_function, 'o-', lw=2, ms=8, color=color,
                 label=f"{label} (N={sum(number_counts):,})")
        
        # Annotate with number counts
        for j, (m, mf, count) in enumerate(zip(bin_centers, mass_function, number_counts)):
            ax.annotate(f"{count:,}", (m, mf), xytext=(0, 10), 
                       textcoords='offset points', ha='center', fontsize=9, color=color)
    
    # Add theoretical mass function if requested
    if add_theory:
        try:
            from hmf import MassFunction
            
            # Get mass range from data
            min_mass = min([min(data['bin_centers']) for data in all_data])
            max_mass = max([max(data['bin_centers']) for data in all_data])
            
            # Create a finer mass grid for the theoretical curve
            mass_grid = np.logspace(np.log10(min_mass*0.5), np.log10(max_mass*2), 100)
            
            # Create mass function objects for each cosmology
            cosmo_params = [
                {'sigma_8': SIGMA8_G, 'label': 'Gaussian'},  # Gaussian
                {'sigma_8': SIGMA8_H, 'label': 'High σ₈'},   # High σ8
                {'sigma_8': SIGMA8_L, 'label': 'Low σ₈'}     # Low σ8
            ]
            
            for i, params in enumerate(cosmo_params):
                hmf = MassFunction(z=z_value, Mmin=np.log10(mass_grid[0]), 
                                  Mmax=np.log10(mass_grid[-1]), 
                                  dlog10m=0.01, sigma_8=params['sigma_8'])
                
                # Get the mass function (dn/dlog10M)
                mf = hmf.dndlog10m
                
                # Plot the theoretical curve
                color = colors[i % len(colors)]
                ax.loglog(hmf.m, mf, '--', color=color, lw=1.5, 
                         label=f"Theory: {params['label']}")
                
        except ImportError:
            print("Warning: hmf package not installed. Skipping theoretical mass function.")
            print("To install: pip install hmf")
    
    # Set axes
    ax.set_xlabel(r"Halo Mass $M$ [M$_\odot$/h]", fontsize=14)
    ax.set_ylabel(r"dn/dlog$M$ [(Mpc/h)$^{-3}$]", fontsize=14)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(f"Halo Mass Function Comparison (z={z_value}, snap={snap})", fontsize=16)
    
    # Add grid and legend
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=12)
    
    # Create ratio plot if we have multiple simulations
    if len(all_mass_functions) > 1:
        # Create a new figure for the ratio plot
        fig_ratio, ax_ratio = plt.subplots(figsize=figsize)
        
        # Use the first simulation as reference
        ref_centers = all_bin_centers[0]
        ref_mf = all_mass_functions[0]
        ref_label = all_labels[0]
        
        # Plot ratios
        for i in range(1, len(all_mass_functions)):
            centers = all_bin_centers[i]
            mf = all_mass_functions[i]
            label = all_labels[i]
            
            # Compute ratio
            ratio = mf / ref_mf
            
            # Plot
            color = colors[i % len(colors)]
            ax_ratio.semilogx(centers, ratio, 'o-', lw=2, ms=8, color=color,
                            label=f"{label} / {ref_label}")
        
        # Set axes
        ax_ratio.set_xlabel(r"Halo Mass $M$ [M$_\odot$/h]", fontsize=14)
        ax_ratio.set_ylabel("Mass Function Ratio", fontsize=14)
        ax_ratio.set_title(f"Ratio to Reference Simulation (z={z_value})", fontsize=16)
        
        # Add grid and legend
        ax_ratio.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_ratio.legend(frameon=True, fontsize=12)
        ax_ratio.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        
        # Save ratio plot
        ratio_file = os.path.splitext(output_file)[0] + "_ratio.png" if output_file else f"mass_function_ratio_snap{snap}.png"
        plt.figure(fig_ratio.number)
        plt.tight_layout()
        plt.savefig(ratio_file, dpi=dpi, bbox_inches='tight')
        print(f"Ratio plot saved to: {ratio_file}")
    
    # Save main figure
    if output_file is None:
        output_file = f"mass_function_comparison_snap{snap}.png"
    
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    
    return fig, ax

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main execution function"""
    # Find mass function data files
    data_files = find_mass_function_files(DATA_DIR, snap=SNAP)
    
    if not data_files:
        print(f"No mass function data files found for snapshot {SNAP}")
        return
    
    # Set output file name
    z_value = 0 if SNAP==624 else 1 if SNAP==310 else 2 if SNAP==205 else "unknown"
    output_file = f"mass_function_plot_z{z_value}_snap{SNAP}.png"
    
    # Plot mass functions
    plot_mass_functions(
        data_files,
        output_file=output_file,
        title=PLOT_TITLE,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        dpi=FIGURE_DPI,
        add_theory=ADD_THEORY
    )
    
    # Print summary of files used
    print("\nFiles used for plotting:")
    for file in data_files:
        print(f"- {file}")

if __name__ == "__main__":
    main()