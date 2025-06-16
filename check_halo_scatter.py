#!/usr/bin/env python3
"""
Create scatter plots to check for obvious issues in Gaussian simulation halo data.
Visualize halo properties including mass, position, and other characteristics.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Add this for 3D plotting
import pygio
import os
from pathlib import Path

# Configuration
SNAP = "624"
BASE_GAUSS = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/")
OUTPUT_DIR = "./halo_check_plots"

# Simulation box size
BOXSIZE = 2000.0  # Mpc/h

def read_halo_data():
    """Read halo data from all files"""
    halo_file = f"{BASE_GAUSS}/HALOS-b0168/m000-{SNAP}.haloproperties"
    
    print(f"Reading halo data from: {halo_file}")
    
    all_data = {}
    successful_files = 0
    
    # Read first file to get available keys
    try:
        data_0 = pygio.read_genericio(f"{halo_file}#0")
        print(f"Available data fields: {list(data_0.keys())}")
        
        # Initialize data dictionary
        for key in data_0.keys():
            all_data[key] = []
            
    except Exception as e:
        print(f"Error reading first file: {e}")
        return None
    
    # Read all 256 files
    for i in range(256):
        try:
            data = pygio.read_genericio(f"{halo_file}#{i}")
            for key in data.keys():
                all_data[key].extend(data[key])
            successful_files += 1
            
            if i % 50 == 0:  # Progress indicator
                print(f"Read {i+1}/256 files...")
                
        except Exception as e:
            print(f"Warning: Cannot read file #{i}: {e}")
            continue
    
    print(f"Successfully read {successful_files}/256 files")
    
    # Convert to numpy arrays
    for key in all_data.keys():
        all_data[key] = np.array(all_data[key])
    
    # Only print summary for key fields
    print(f"\nKey field summary:")
    print(f"sod_halo_mass: {len(all_data['sod_halo_mass'])} entries, range: {all_data['sod_halo_mass'].min():.3e} to {all_data['sod_halo_mass'].max():.3e}")
    if "sod_halo_center_x" in all_data:
        print(f"sod_halo_center_x: range: {all_data['sod_halo_center_x'].min():.1f} to {all_data['sod_halo_center_x'].max():.1f}")
        print(f"sod_halo_center_y: range: {all_data['sod_halo_center_y'].min():.1f} to {all_data['sod_halo_center_y'].max():.1f}")
        print(f"sod_halo_center_z: range: {all_data['sod_halo_center_z'].min():.1f} to {all_data['sod_halo_center_z'].max():.1f}")
    
    return all_data

def create_scatter_plots(data):
    """Create various scatter plots to check data quality"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract key quantities
    halo_mass = data["sod_halo_mass"]
    
    # Use SOD halo center positions (same as galaxy generation code)
    if "sod_halo_center_x" in data and "sod_halo_center_y" in data and "sod_halo_center_z" in data:
        halo_x = data["sod_halo_center_x"]
        halo_y = data["sod_halo_center_y"]
        halo_z = data["sod_halo_center_z"]
        position_fields = ["sod_halo_center_x", "sod_halo_center_y", "sod_halo_center_z"]
    else:
        print("Error: Cannot find SOD halo center position fields in data!")
        print("Available fields:", list(data.keys()))
        return
    
    print(f"Using position fields: {position_fields}")
    
    # Filter positive mass halos
    positive_mass_mask = halo_mass > 0
    halo_mass_pos = halo_mass[positive_mass_mask]
    halo_x_pos = halo_x[positive_mass_mask]
    halo_y_pos = halo_y[positive_mass_mask]
    halo_z_pos = halo_z[positive_mass_mask]
    
    print(f"\nHalo statistics:")
    print(f"Total halos: {len(halo_mass):,}")
    print(f"Positive mass halos: {len(halo_mass_pos):,}")
    print(f"Mass range: {halo_mass_pos.min():.3e} to {halo_mass_pos.max():.3e} M☉/h")
    
    # 1. Mass histogram
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(halo_mass_pos), bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=np.log10(1e11), color='red', linestyle='--', label='1×10¹¹ M☉/h')
    plt.axvline(x=np.log10(3.16e11), color='red', linestyle=':', label='3.16×10¹¹ M☉/h')
    plt.axvline(x=np.log10(1e12), color='purple', linestyle='-.', label='1×10¹² M☉/h')
    plt.xlabel('log₁₀(Halo Mass [M☉/h])')
    plt.ylabel('Number of Halos')
    plt.title('Halo Mass Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mass_histogram.png", dpi=300)
    plt.close()
    
    # 2. Spatial distribution (2D projections)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sample halos for plotting (use subset for large datasets)
    n_sample = min(10000, len(halo_mass_pos))
    sample_idx = np.random.choice(len(halo_mass_pos), n_sample, replace=False)
    
    x_sample = halo_x_pos[sample_idx]
    y_sample = halo_y_pos[sample_idx]
    z_sample = halo_z_pos[sample_idx]
    mass_sample = halo_mass_pos[sample_idx]
    
    # Color by mass
    scatter_kwargs = {'c': np.log10(mass_sample), 'cmap': 'viridis', 's': 1, 'alpha': 0.6}
    
    # X-Y projection
    im1 = axes[0].scatter(x_sample, y_sample, **scatter_kwargs)
    axes[0].set_xlabel('sod_halo_center_x [Mpc/h]')
    axes[0].set_ylabel('sod_halo_center_y [Mpc/h]')
    axes[0].set_title('X-Y Projection')
    axes[0].set_xlim(0, BOXSIZE)
    axes[0].set_ylim(0, BOXSIZE)
    
    # X-Z projection  
    im2 = axes[1].scatter(x_sample, z_sample, **scatter_kwargs)
    axes[1].set_xlabel('sod_halo_center_x [Mpc/h]')
    axes[1].set_ylabel('sod_halo_center_z [Mpc/h]')
    axes[1].set_title('X-Z Projection')
    axes[1].set_xlim(0, BOXSIZE)
    axes[1].set_ylim(0, BOXSIZE)
    
    # Y-Z projection
    im3 = axes[2].scatter(y_sample, z_sample, **scatter_kwargs)
    axes[2].set_xlabel('sod_halo_center_y [Mpc/h]')
    axes[2].set_ylabel('sod_halo_center_z [Mpc/h]')
    axes[2].set_title('Y-Z Projection')
    axes[2].set_xlim(0, BOXSIZE)
    axes[2].set_ylim(0, BOXSIZE)
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=axes, shrink=0.8)
    cbar.set_label('log₁₀(Halo Mass [M☉/h])')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spatial_distribution.png", dpi=300)
    plt.close()
    
    # 2b. 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use smaller sample for 3D plot (can be slow)
    n_sample_3d = min(5000, len(halo_mass_pos))
    sample_idx_3d = np.random.choice(len(halo_mass_pos), n_sample_3d, replace=False)
    
    x_3d = halo_x_pos[sample_idx_3d]
    y_3d = halo_y_pos[sample_idx_3d]
    z_3d = halo_z_pos[sample_idx_3d]
    mass_3d = halo_mass_pos[sample_idx_3d]
    
    # Color by mass, size by mass
    colors = np.log10(mass_3d)
    sizes = (mass_3d / 1e11) * 5  # Scale size with mass
    sizes = np.clip(sizes, 1, 50)  # Limit size range
    
    scatter = ax.scatter(x_3d, y_3d, z_3d, c=colors, s=sizes, 
                        cmap='viridis', alpha=0.6)
    
    ax.set_xlabel('sod_halo_center_x [Mpc/h]')
    ax.set_ylabel('sod_halo_center_y [Mpc/h]')
    ax.set_zlabel('sod_halo_center_z [Mpc/h]')
    ax.set_title('3D Halo Distribution (colored and sized by mass)')
    
    # Set equal aspect ratio
    ax.set_xlim(0, BOXSIZE)
    ax.set_ylim(0, BOXSIZE)
    ax.set_zlim(0, BOXSIZE)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, shrink=0.5, aspect=20)
    cbar.set_label('log₁₀(Halo Mass [M☉/h])')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spatial_distribution_3d.png", dpi=300)
    plt.close()
    
    # 2c. Sliced projections
    # Define slice positions (center regions of the box)
    slice_thickness = 50.0  # Mpc/h
    slice_centers = [BOXSIZE/4, BOXSIZE/2, 3*BOXSIZE/4]  # Three slices
    
    # X-slices (fix X, show Y-Z)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, x_center in enumerate(slice_centers):
        x_min, x_max = x_center - slice_thickness/2, x_center + slice_thickness/2
        slice_mask = (halo_x_pos >= x_min) & (halo_x_pos <= x_max)
        
        if np.sum(slice_mask) > 0:
            y_slice = halo_y_pos[slice_mask]
            z_slice = halo_z_pos[slice_mask]
            mass_slice = halo_mass_pos[slice_mask]
            
            # Sample if too many points
            if len(y_slice) > 5000:
                sample_idx = np.random.choice(len(y_slice), 5000, replace=False)
                y_slice = y_slice[sample_idx]
                z_slice = z_slice[sample_idx]
                mass_slice = mass_slice[sample_idx]
            
            scatter = axes[i].scatter(y_slice, z_slice, c=np.log10(mass_slice), 
                                    cmap='viridis', s=2, alpha=0.7)
            axes[i].set_xlabel('sod_halo_center_y [Mpc/h]')
            axes[i].set_ylabel('sod_halo_center_z [Mpc/h]')
            axes[i].set_title(f'X-slice: {x_center:.0f}±{slice_thickness/2:.0f} Mpc/h\n(n={np.sum(slice_mask):,})')
            axes[i].set_xlim(0, BOXSIZE)
            axes[i].set_ylim(0, BOXSIZE)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No halos in slice', transform=axes[i].transAxes, 
                        ha='center', va='center')
            axes[i].set_title(f'X-slice: {x_center:.0f}±{slice_thickness/2:.0f} Mpc/h')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/x_sliced_projections.png", dpi=300)
    plt.close()
    
    # Y-slices (fix Y, show X-Z)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, y_center in enumerate(slice_centers):
        y_min, y_max = y_center - slice_thickness/2, y_center + slice_thickness/2
        slice_mask = (halo_y_pos >= y_min) & (halo_y_pos <= y_max)
        
        if np.sum(slice_mask) > 0:
            x_slice = halo_x_pos[slice_mask]
            z_slice = halo_z_pos[slice_mask]
            mass_slice = halo_mass_pos[slice_mask]
            
            # Sample if too many points
            if len(x_slice) > 5000:
                sample_idx = np.random.choice(len(x_slice), 5000, replace=False)
                x_slice = x_slice[sample_idx]
                z_slice = z_slice[sample_idx]
                mass_slice = mass_slice[sample_idx]
            
            scatter = axes[i].scatter(x_slice, z_slice, c=np.log10(mass_slice), 
                                    cmap='viridis', s=2, alpha=0.7)
            axes[i].set_xlabel('sod_halo_center_x [Mpc/h]')
            axes[i].set_ylabel('sod_halo_center_z [Mpc/h]')
            axes[i].set_title(f'Y-slice: {y_center:.0f}±{slice_thickness/2:.0f} Mpc/h\n(n={np.sum(slice_mask):,})')
            axes[i].set_xlim(0, BOXSIZE)
            axes[i].set_ylim(0, BOXSIZE)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No halos in slice', transform=axes[i].transAxes, 
                        ha='center', va='center')
            axes[i].set_title(f'Y-slice: {y_center:.0f}±{slice_thickness/2:.0f} Mpc/h')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/y_sliced_projections.png", dpi=300)
    plt.close()
    
    # Z-slices (fix Z, show X-Y)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, z_center in enumerate(slice_centers):
        z_min, z_max = z_center - slice_thickness/2, z_center + slice_thickness/2
        slice_mask = (halo_z_pos >= z_min) & (halo_z_pos <= z_max)
        
        if np.sum(slice_mask) > 0:
            x_slice = halo_x_pos[slice_mask]
            y_slice = halo_y_pos[slice_mask]
            mass_slice = halo_mass_pos[slice_mask]
            
            # Sample if too many points
            if len(x_slice) > 5000:
                sample_idx = np.random.choice(len(x_slice), 5000, replace=False)
                x_slice = x_slice[sample_idx]
                y_slice = y_slice[sample_idx]
                mass_slice = mass_slice[sample_idx]
            
            scatter = axes[i].scatter(x_slice, y_slice, c=np.log10(mass_slice), 
                                    cmap='viridis', s=2, alpha=0.7)
            axes[i].set_xlabel('sod_halo_center_x [Mpc/h]')
            axes[i].set_ylabel('sod_halo_center_y [Mpc/h]')
            axes[i].set_title(f'Z-slice: {z_center:.0f}±{slice_thickness/2:.0f} Mpc/h\n(n={np.sum(slice_mask):,})')
            axes[i].set_xlim(0, BOXSIZE)
            axes[i].set_ylim(0, BOXSIZE)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No halos in slice', transform=axes[i].transAxes, 
                        ha='center', va='center')
            axes[i].set_title(f'Z-slice: {z_center:.0f}±{slice_thickness/2:.0f} Mpc/h')
    
    # Add colorbar to the last figure
    cbar = plt.colorbar(scatter, ax=axes, shrink=0.8)
    cbar.set_label('log₁₀(Halo Mass [M☉/h])')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/z_sliced_projections.png", dpi=300)
    plt.close()
    
    # 3. Check for edge effects or clustering issues
    plt.figure(figsize=(12, 4))
    
    # Position histograms
    plt.subplot(1, 3, 1)
    plt.hist(halo_x_pos, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('sod_halo_center_x Position [Mpc/h]')
    plt.ylabel('Number of Halos')
    plt.title('X Position Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(halo_y_pos, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('sod_halo_center_y Position [Mpc/h]')
    plt.ylabel('Number of Halos')
    plt.title('Y Position Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(halo_z_pos, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('sod_halo_center_z Position [Mpc/h]')
    plt.ylabel('Number of Halos')
    plt.title('Z Position Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/position_distributions.png", dpi=300)
    plt.close()
    
    # 4. Mass vs radius (if radius data available)
    if "sod_halo_radius" in data:
        halo_radius = data["sod_halo_radius"][positive_mass_mask]
        
        plt.figure(figsize=(10, 8))
        
        # Sample for scatter plot
        sample_for_plot = slice(None, None, 10)  # Every 10th point
        plt.scatter(np.log10(halo_mass_pos[sample_for_plot]), 
                   np.log10(halo_radius[sample_for_plot]), 
                   alpha=0.5, s=1)
        plt.xlabel('log₁₀(Halo Mass [M☉/h])')
        plt.ylabel('log₁₀(sod_halo_radius [Mpc/h])')
        plt.title('Mass-Radius Relation')
        plt.grid(True, alpha=0.3)
        
        # Add expected scaling line (R ∝ M^(1/3))
        mass_range = np.linspace(np.log10(halo_mass_pos.min()), np.log10(halo_mass_pos.max()), 100)
        # Normalize to match data roughly
        radius_expected = (mass_range - 11) / 3 + np.log10(0.5)  # Rough scaling
        plt.plot(mass_range, radius_expected, 'r--', alpha=0.8, label='R ∝ M^(1/3) scaling')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/mass_radius_relation.png", dpi=300)
        plt.close()
    else:
        print("No sod_halo_radius field found in data")
    
    # 5. Check for specific mass ranges of interest
    plt.figure(figsize=(12, 8))
    
    # Define mass bins
    mass_bins = [
        (1e11, 3.16e11, "1e11-3.16e11"),
        (3.16e11, 1e12, "3.16e11-1e12"),
        (1e12, 3.16e12, "1e12-3.16e12")
    ]
    
    colors = ['red', 'blue', 'green']
    
    for i, (mass_min, mass_max, label) in enumerate(mass_bins):
        mask = (halo_mass_pos >= mass_min) & (halo_mass_pos <= mass_max)
        if np.sum(mask) > 0:
            # Sample for visualization
            sample_size = min(5000, np.sum(mask))
            sample_idx = np.random.choice(np.where(mask)[0], sample_size, replace=False)
            
            plt.scatter(halo_x_pos[sample_idx], halo_y_pos[sample_idx], 
                       c=colors[i], alpha=0.6, s=2, label=f'{label} M☉/h (n={np.sum(mask):,})')
    
    plt.xlabel('sod_halo_center_x [Mpc/h]')
    plt.ylabel('sod_halo_center_y [Mpc/h]')
    plt.title('Halo Distribution by Mass Range')
    plt.legend()
    plt.xlim(0, BOXSIZE)
    plt.ylim(0, BOXSIZE)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mass_range_distribution.png", dpi=300)
    plt.close()
    
    # 6. Check for obvious data issues
    print(f"\nData quality checks:")
    print(f"sod_halo_center_x position range: {halo_x_pos.min():.3f} to {halo_x_pos.max():.3f}")
    print(f"sod_halo_center_y position range: {halo_y_pos.min():.3f} to {halo_y_pos.max():.3f}")
    print(f"sod_halo_center_z position range: {halo_z_pos.min():.3f} to {halo_z_pos.max():.3f}")
    
    # Check for halos outside box
    outside_box = (halo_x_pos < 0) | (halo_x_pos > BOXSIZE) | \
                  (halo_y_pos < 0) | (halo_y_pos > BOXSIZE) | \
                  (halo_z_pos < 0) | (halo_z_pos > BOXSIZE)
    print(f"Halos outside box boundaries: {np.sum(outside_box):,}")
    
    # Check for duplicate positions
    positions = np.column_stack([halo_x_pos, halo_y_pos, halo_z_pos])
    unique_positions = np.unique(positions, axis=0)
    print(f"Unique positions: {len(unique_positions):,} out of {len(positions):,}")
    
    # Mass range statistics for our analysis
    for mass_min, mass_max, label in mass_bins:
        mask = (halo_mass_pos >= mass_min) & (halo_mass_pos <= mass_max)
        count = np.sum(mask)
        density = count / (BOXSIZE**3)
        print(f"{label} M☉/h: {count:,} halos, density = {density:.6e} (h/Mpc)³")

def main():
    """Main function"""
    print("Checking Gaussian simulation halo data for obvious issues")
    print("=" * 60)
    
    # Read halo data
    data = read_halo_data()
    if data is None:
        print("Failed to read halo data")
        return
    
    # Create scatter plots and checks
    create_scatter_plots(data)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("Check the plots for:")
    print("- Uniform spatial distribution")
    print("- Reasonable mass distribution")
    print("- No obvious clustering artifacts")
    print("- Proper boundary conditions")
    print("\nProcessing completed!")

if __name__ == "__main__":
    main() 