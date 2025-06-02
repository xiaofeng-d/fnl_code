#!/usr/bin/env python3
"""
Compute and plot halo mass function from simulation data.
This script reads halo catalogs and computes the mass function for different simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
import argparse
from pathlib import Path
import re
import sys

try:
    import pygio  # type: ignore
except ImportError:
    sys.exit("pygio not installed – `pip install pygio` required to read .haloproperties files.")

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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SNAP = 624  # ROCKSTAR step number (624=z0, 310=z1, 205=z2)

# Directories (each contains a HALOS-b0168 subdir with .haloproperties files)
BASE_GAUSS = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0")
BASE_S8H   = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0_s8h")
BASE_S8L   = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0_s8l")

# Simulation parameters
BOX_SIZE = 2000.0  # Mpc/h
SIGMA8_G = 0.834   # baseline σ₈
SIGMA8_H = 0.844   # high‑σ₈ value
SIGMA8_L = 0.824   # low‑σ₈ value

# Mass bins for analysis (M⊙/h)
MASS_BINS = [
    (1.00e11, 3.16e11),
    (3.16e11, 1.00e12),
    (1.00e12, 3.16e12),
    (3.16e12, 1.00e13),
    (1.00e13, 3.16e13),
    (3.16e13, 1.00e14),
    (1.00e14, 3.16e14),
    (3.16e14, 1.00e15),
    (1.00e15, 1.00e16),
]

# ─────────────────────────────────────────────────────────────────────────────

def halos_dir(base: Path) -> Path:
    """Return the path to the HALOS directory"""
    return base / "HALOS-b0168"

def format_mass(m):
    """Format mass value, removing plus sign from scientific notation"""
    return f"{m:.2e}".replace('+', '')

def mass_str(m):
    """Format mass for filenames, removing decimal point and exponent sign"""
    return format_mass(m).replace('.', 'p').replace('+', '')

def read_halo_masses(base_dir, snap):
    """
    Read halo masses from .haloproperties files
    
    Parameters:
        base_dir: Base directory containing HALOS-b0168
        snap: Snapshot number
        
    Returns:
        Array of halo masses (M⊙/h)
    """
    hdir = halos_dir(base_dir)
    print(f"Reading halos from {hdir}")
    
    # Find all .haloproperties files for this snapshot
    pattern = f"m???-{snap}.haloproperties"
    files = sorted(glob.glob(str(hdir / pattern)))
    
    if not files:
        print(f"Warning: No haloproperties files found matching {hdir / pattern}")
        return np.array([])
    
    print(f"Found {len(files)} haloproperties files")
    
    # Read and concatenate masses from all files
    all_masses = []
    
    for f in files:
        try:
            # Use read_genericio as shown in the example
            data = pygio.read_genericio(f)
            masses = data['sod_halo_mass']
            all_masses.append(masses)
            print(f"  {Path(f).name}: {len(masses)} halos")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not all_masses:
        return np.array([])
    
    return np.concatenate(all_masses)

def compute_mass_function(masses, mass_bins, box_size):
    """
    Compute the halo mass function
    
    Parameters:
        masses: Array of halo masses
        mass_bins: List of (min_mass, max_mass) tuples
        box_size: Simulation box size in Mpc/h
        
    Returns:
        bin_centers, mass_function, number_counts
    """
    volume = box_size**3  # (Mpc/h)^3
    
    bin_centers = []
    mass_function = []
    number_counts = []
    
    for ml, mh in mass_bins:
        # Count halos in this mass bin
        mask = (masses >= ml) & (masses < mh)
        count = np.sum(mask)
        
        # Compute bin center (geometric mean)
        bin_center = np.sqrt(ml * mh)
        
        # Compute mass function (dn/dlogM)
        # This is number density per log mass interval
        log_mass_interval = np.log10(mh) - np.log10(ml)
        mf = count / volume / log_mass_interval
        
        bin_centers.append(bin_center)
        mass_function.append(mf)
        number_counts.append(count)
    
    return np.array(bin_centers), np.array(mass_function), np.array(number_counts)

def plot_mass_function(bin_centers, mass_function, number_counts, 
                      output_file=None, title=None, figsize=(10, 8), dpi=300, snap=SNAP):
    """
    Plot the halo mass function
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mass function
    ax.loglog(bin_centers, mass_function, 'o-', lw=2, ms=8, color='blue',
             label=f"Mass Function (N={sum(number_counts)})")
    
    # Annotate with number counts
    for i, (m, mf, count) in enumerate(zip(bin_centers, mass_function, number_counts)):
        ax.annotate(f"{count}", (m, mf), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=9)
    
    # Set axes
    ax.set_xlabel(r"Halo Mass $M$ [M$_\odot$/h]", fontsize=14)
    ax.set_ylabel(r"dn/dlog$M$ [(Mpc/h)$^{-3}$]", fontsize=14)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=16)
    else:
        z_value = 0 if snap==624 else 1 if snap==310 else 2 if snap==205 else "?"
        ax.set_title(f"Halo Mass Function (z={z_value}, snap={snap})", fontsize=16)
    
    # Add grid and legend
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=12)
    
    # Save figure
    if output_file is None:
        output_file = f"mass_function_snap{snap}.png"
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    return fig, ax

def save_mass_function_data(bin_centers, mass_function, number_counts, sim_name, snap, output_dir=None):
    """
    Save mass function data to a numpy .npz file
    
    Parameters:
        bin_centers: Array of mass bin centers
        mass_function: Array of mass function values
        number_counts: Array of halo counts per bin
        sim_name: Simulation name (e.g., 'gauss', 's8h', 's8l')
        snap: Snapshot number
        output_dir: Directory to save the file (default: current directory)
    
    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = "."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    z_value = 0 if snap==624 else 1 if snap==310 else 2 if snap==205 else "unknown"
    filename = f"mass_function_{sim_name}_z{z_value}_snap{snap}.npz"
    filepath = os.path.join(output_dir, filename)
    
    # Save data
    np.savez(
        filepath,
        bin_centers=bin_centers,
        mass_function=mass_function,
        number_counts=number_counts,
        mass_bins=MASS_BINS,
        box_size=BOX_SIZE,
        snap=snap,
        sim_name=sim_name,
        z=z_value
    )
    
    print(f"Mass function data saved to: {filepath}")
    return filepath

def compare_mass_functions(base_dirs, labels, output_file=None, title=None, 
                          figsize=(10, 8), dpi=300, snap=SNAP, add_theory=False,
                          save_data=False, output_dir=None):
    """
    Compare mass functions from different simulations
    
    Parameters:
        base_dirs: List of base directories for simulations
        labels: List of labels for simulations
        output_file: Output file path
        title: Plot title
        figsize: Figure size
        dpi: Figure resolution
        snap: Snapshot number
        add_theory: Whether to add theoretical mass function predictions
        save_data: Whether to save mass function data to files
        output_dir: Directory to save data files
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Store data for ratio plot
    all_bin_centers = []
    all_mass_functions = []
    
    # Process each simulation
    for i, (base_dir, label) in enumerate(zip(base_dirs, labels)):
        # Extract simulation name from label or base_dir
        if "Gaussian" in label:
            sim_name = "gauss"
        elif "High" in label:
            sim_name = "s8h"
        elif "Low" in label:
            sim_name = "s8l"
        else:
            sim_name = f"sim{i}"
        
        # Read halo masses
        masses = read_halo_masses(base_dir, snap)
        
        if len(masses) == 0:
            print(f"Warning: No halos found for {label}")
            continue
        
        # Compute mass function
        bin_centers, mass_function, number_counts = compute_mass_function(
            masses, MASS_BINS, BOX_SIZE)
        
        # Save data if requested
        if save_data:
            save_mass_function_data(
                bin_centers, mass_function, number_counts, 
                sim_name, snap, output_dir
            )
        
        # Store data
        all_bin_centers.append(bin_centers)
        all_mass_functions.append(mass_function)
        
        # Plot mass function
        color = colors[i % len(colors)]
        ax.loglog(bin_centers, mass_function, 'o-', lw=2, ms=8, color=color,
                 label=f"{label} (N={len(masses):,})")
        
        # Annotate with number counts
        for j, (m, mf, count) in enumerate(zip(bin_centers, mass_function, number_counts)):
            ax.annotate(f"{count:,}", (m, mf), xytext=(0, 10), 
                       textcoords='offset points', ha='center', fontsize=9, color=color)
    
    # Add theoretical mass function if requested
    if add_theory:
        # Create a finer mass grid for the theoretical curve
        mass_grid = np.logspace(np.log10(MASS_BINS[0][0]), np.log10(MASS_BINS[-1][1]), 100)
        
        # Add Sheth-Tormen mass function
        try:
            from hmf import MassFunction
            
            # Get redshift from snapshot
            z_value = 0 if snap==624 else 1 if snap==310 else 2 if snap==205 else 0
            
            # Create mass function objects for each cosmology
            cosmo_params = [
                {'sigma_8': SIGMA8_G},  # Gaussian
                {'sigma_8': SIGMA8_H},  # High σ8
                {'sigma_8': SIGMA8_L}   # Low σ8
            ]
            
            for i, params in enumerate(cosmo_params):
                hmf = MassFunction(z=z_value, Mmin=np.log10(mass_grid[0]), 
                                  Mmax=np.log10(mass_grid[-1]), 
                                  dlog10m=0.01, **params)
                
                # Get the mass function (dn/dlog10M)
                mf = hmf.dndlog10m
                
                # Plot the theoretical curve
                color = colors[i % len(colors)]
                ax.loglog(hmf.m, mf, '--', color=color, lw=1.5, 
                         label=f"Theory (σ₈={params['sigma_8']})")
                
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
        z_value = 0 if snap==624 else 1 if snap==310 else 2 if snap==205 else "?"
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
        
        # Plot ratios
        for i in range(1, len(all_mass_functions)):
            centers = all_bin_centers[i]
            mf = all_mass_functions[i]
            
            # Compute ratio
            ratio = mf / ref_mf
            
            # Plot
            color = colors[i % len(colors)]
            ax_ratio.semilogx(centers, ratio, 'o-', lw=2, ms=8, color=color,
                            label=f"{labels[i]} / {labels[0]}")
        
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

def main():
    parser = argparse.ArgumentParser(description="Compute and plot halo mass function")
    parser.add_argument('--snap', type=int, default=SNAP, 
                        help=f'Snapshot number (default: {SNAP} for z=0)')
    parser.add_argument('--output', '-o', help='Output image file path')
    parser.add_argument('--title', '-t', help='Plot title')
    parser.add_argument('--width', type=float, default=10, help='Figure width')
    parser.add_argument('--height', type=float, default=8, help='Figure height')
    parser.add_argument('--dpi', type=int, default=300, help='Figure resolution')
    parser.add_argument('--single', action='store_true', 
                        help='Plot only the Gaussian simulation (default: compare all)')
    parser.add_argument('--theory', action='store_true',
                        help='Add theoretical mass function predictions')
    parser.add_argument('--save-data', action='store_true',
                        help='Save mass function data to .npz files')
    parser.add_argument('--output-dir', default='mass_function_data',
                        help='Directory to save data files (default: mass_function_data)')
    args = parser.parse_args()
    
    snap = args.snap  # Use local variable instead of modifying global
    
    if args.single:
        # Compute and plot mass function for a single simulation
        masses = read_halo_masses(BASE_GAUSS, snap)
        
        if len(masses) == 0:
            print("Error: No halos found")
            return
        
        bin_centers, mass_function, number_counts = compute_mass_function(
            masses, MASS_BINS, BOX_SIZE)
        
        # Save data if requested
        if args.save_data:
            save_mass_function_data(
                bin_centers, mass_function, number_counts, 
                "gauss", snap, args.output_dir
            )
        
        plot_mass_function(
            bin_centers, 
            mass_function, 
            number_counts,
            snap=snap,
            output_file=args.output,
            title=args.title,
            figsize=(args.width, args.height),
            dpi=args.dpi
        )
        
        # Print mass function data
        print("\nMass Function Data:")
        print(f"{'Mass Range [M⊙/h]':30s} {'Count':8s} {'dn/dlogM [(Mpc/h)^-3]':20s}")
        print("-" * 60)
        
        for i, (ml, mh) in enumerate(MASS_BINS):
            mass_range = f"{format_mass(ml)}-{format_mass(mh)}"
            count = number_counts[i]
            mf = mass_function[i]
            print(f"{mass_range:30s} {count:8d} {mf:20.6e}")
    else:
        # Compare mass functions from different simulations (default behavior)
        compare_mass_functions(
            [BASE_GAUSS, BASE_S8H, BASE_S8L],
            [f"Gaussian (σ₈={SIGMA8_G})", f"High σ₈ ({SIGMA8_H})", f"Low σ₈ ({SIGMA8_L})"],
            snap=snap,
            output_file=args.output,
            title=args.title,
            figsize=(args.width, args.height),
            dpi=args.dpi,
            add_theory=args.theory,
            save_data=args.save_data,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
