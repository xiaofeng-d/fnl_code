#!/usr/bin/env python3
"""
Plot k*P(k) for galaxy power spectrum from HACC simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def load_power_spectrum(filename):
    """Load power spectrum data from file."""
    # Skip the header line
    data = np.loadtxt(filename, skiprows=1)
    k = data[:, 0]      # Wavenumber (h/Mpc)
    P = data[:, 1]      # Monopole P0(k) (Mpc/h)^3
    err = data[:, 2]    # Error bars
    nmodes = data[:, 3] # Number of modes
    P2 = data[:, 4]     # Quadrupole P2(k) if available
    return k, P, err, nmodes, P2

def plot_kPk():
    """Plot k*P(k) for galaxy power spectrum and calculate bias"""
    
    # File paths
    galaxy_pk_file = '/home/ac.xdong/hacc-bispec/output_l2000n4096_gauss_tpm_seed0/pk.WITHSAT.310.pk'
    halo_pk_file = '/home/ac.xdong/hacc-bispec/output_l2000n4096_gauss_tpm_seed0/pk.1e11-1e13.310.pk'
    matter_pk_file = '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/POWER/m000.pk.310'
    
    # Check if files exist
    for label, filepath in [("Galaxy", galaxy_pk_file), ("Halo", halo_pk_file), ("Matter", matter_pk_file)]:
        if not os.path.exists(filepath):
            print(f"Error: {label} file {filepath} does not exist!")
            return
    
    # Load data
    print(f"Loading galaxy power spectrum from: {galaxy_pk_file}")
    k_gal, P_gal, err_gal, nmodes_gal, P2_gal = load_power_spectrum(galaxy_pk_file)
    
    print(f"Loading halo power spectrum from: {halo_pk_file}")
    k_halo, P_halo, err_halo, nmodes_halo, P2_halo = load_power_spectrum(halo_pk_file)
    
    print(f"Loading matter power spectrum from: {matter_pk_file}")
    k_matter, P_matter, err_matter, nmodes_matter, P2_matter = load_power_spectrum(matter_pk_file)
    
    # Galaxy number and shot noise calculation
    # From terminal output: 中心星系: 1,444,415 + 卫星星系: 66,182 = 1,510,597
    N_galaxies = (6880637+1374029)#1510597
    print(f"Using galaxy count from simulation output: {N_galaxies:,}")
    print("  - Central galaxies: 1,444,415")
    print("  - Satellite galaxies: 66,182")
    
    # Simulation box parameters
    boxsize = 2000.0  # Mpc/h
    volume = boxsize**3  # (Mpc/h)^3
    
    # Calculate shot noise: P_shot = V / N_g
    shot_noise = volume / N_galaxies
    print(f"Calculated shot noise: {shot_noise:.2e} (Mpc/h)³")
    print(f"Galaxy number density: {N_galaxies/volume:.2e} (h/Mpc)⁻³")
    
    # Remove shot noise from galaxy power spectrum
    P_gal_corrected = P_gal - shot_noise
    
    # Calculate k*P(k) with shot noise removed
    kPk_gal = k_gal * P_gal_corrected
    kPk_gal_err = k_gal * err_gal  # Error propagation for k*P(k)
    kPk_gal_raw = k_gal * P_gal  # Keep original for comparison
    
    # Interpolate matter P(k) to galaxy k-grid for bias calculation
    P_matter_interp = np.interp(k_gal, k_matter, P_matter)
    P_halo_interp = np.interp(k_gal, k_halo, P_halo)
    
    # Calculate bias: b = sqrt(P_tracer / P_matter)
    # Only calculate where both P values are positive
    mask_gal = (P_gal_corrected > 0) & (P_matter_interp > 0)
    mask_halo = (P_halo_interp > 0) & (P_matter_interp > 0)
    
    bias_galaxy = np.sqrt(P_gal_corrected / P_matter_interp)
    bias_halo = np.sqrt(P_halo_interp / P_matter_interp)
    
    print(f"Galaxy bias range: [{bias_galaxy[mask_gal].min():.2f}, {bias_galaxy[mask_gal].max():.2f}]")
    print(f"Halo bias range: [{bias_halo[mask_halo].min():.2f}, {bias_halo[mask_halo].max():.2f}]")
    
    # Create the plot with subplots
    fig = plt.figure(figsize=(12, 15))
    ax1 = plt.subplot(3, 1, 1)  # Independent x-axis
    ax2 = plt.subplot(3, 1, 2)  # Independent first, then share with ax3
    ax3 = plt.subplot(3, 1, 3, sharex=ax2)  # Share x-axis with ax2
    
    # Top panel: k*P(k)
    ax1.errorbar(k_gal, kPk_gal_raw, yerr=kPk_gal_err, fmt='o-', markersize=2, linewidth=1, 
                 alpha=0.5, label='Original k×P(k)', color='gray')
    ax1.errorbar(k_gal, kPk_gal, yerr=kPk_gal_err, fmt='o-', markersize=2, linewidth=1.5, 
                 alpha=0.8, label='Shot noise corrected k×P(k)', color='blue')
    
    # Add horizontal line to show shot noise level
    shot_noise_kPk = k_gal * shot_noise
    ax1.plot(k_gal, shot_noise_kPk, '--', color='red', alpha=0.7, 
             label=f'Shot noise: k×{shot_noise:.1e}')
    
    # Add Findlay et al (2411.12023) comparison data
    findlay_file = '/home/ac.xdong/hacc-bispec/galaxy_pk_plots/2411.12023_high.csv'
    findlay_loaded = False
    try:
        findlay_data = np.loadtxt(findlay_file, delimiter=',', skiprows=1)
        k_findlay = findlay_data[:, 0]
        kaiser_factor = 1.6
        kPk_findlay = findlay_data[:, 1]/kaiser_factor #*0.674
        ax1.plot(k_findlay, kPk_findlay, 's-', color='orange', 
                 linewidth=2, markersize=4, label='Findlay et al (2411.12023), Kaiser factor=1.6 corrected')
        print(f"Successfully loaded Findlay et al data: {len(k_findlay)} data points")
        findlay_loaded = True
    except Exception as e:
        print(f"Warning: Could not load Findlay et al data: {e}")
        print("Continuing without reference curve...")
    
    # Customize top panel
    ax1.set_ylabel(r'$k \times P(k)$ [(Mpc/h)² ]', fontsize=12)
    ax1.set_title('Galaxy Power Spectrum: k×P(k) vs k (Shot Noise Corrected)\n(Gaussian simulation, step 310, z≈1.0)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0.020, 0.200)
    ax1.set_ylim(180, 600)
    
    # Bottom panel: P(k) comparison
    ax2.plot(k_matter, P_matter, '-', color='black', linewidth=2, 
             label='Matter P(k)', alpha=0.8)
    ax2.plot(k_halo, P_halo, 's-', color='red', linewidth=2, 
             markersize=3, label='Halo P(k)', alpha=0.8)
    ax2.plot(k_gal, P_gal_corrected, 'o-', color='blue', linewidth=2, 
             markersize=3, label='Galaxy P(k) (shot noise corrected)', alpha=0.8)
    # Add Findlay P(k) if loaded
    if findlay_loaded:
        # Avoid division by zero
        mask_nonzero = k_findlay > 0
        Pk_findlay = np.zeros_like(k_findlay)
        Pk_findlay[mask_nonzero] = kPk_findlay[mask_nonzero] / k_findlay[mask_nonzero]
        ax2.plot(k_findlay[mask_nonzero], Pk_findlay[mask_nonzero], 's-', color='orange', linewidth=2, markersize=4, label='Findlay et al P(k)')
    
    # Customize bottom panel
    ax2.set_xlabel(r'$k$ [h/Mpc]', fontsize=12)
    ax2.set_ylabel(r'$P(k)$ [(Mpc/h)³]', fontsize=12)
    ax2.set_title('Power Spectrum Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0.020, 0.200)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1e3, 2e4)
    
    # Third panel: Bias comparison
    ax3.plot(k_gal[mask_gal], bias_galaxy[mask_gal], 'o-', color='blue', linewidth=2, 
             markersize=3, label='Galaxy bias', alpha=0.8)
    ax3.plot(k_gal[mask_halo], bias_halo[mask_halo], 's-', color='red', linewidth=2, 
             markersize=3, label='Halo bias', alpha=0.8)
    # Add Findlay bias if loaded
    if findlay_loaded:
        # Compute Findlay P(k)
        mask_nonzero = k_findlay > 0
        Pk_findlay = np.zeros_like(k_findlay)
        Pk_findlay[mask_nonzero] = kPk_findlay[mask_nonzero] / k_findlay[mask_nonzero]
        # Interpolate matter P(k) to Findlay k grid
        P_matter_findlay = np.interp(k_findlay, k_matter, P_matter)
        # Compute bias where both are positive
        mask_bias = (Pk_findlay > 0) & (P_matter_findlay > 0)
        bias_findlay = np.sqrt(Pk_findlay[mask_bias] / P_matter_findlay[mask_bias])
        ax3.plot(k_findlay[mask_bias], bias_findlay, 's-', color='orange', linewidth=2, markersize=4, label='Findlay et al bias')
    
    # Customize third panel
    ax3.set_xlabel(r'$k$ [h/Mpc]', fontsize=12)
    ax3.set_ylabel(r'$b(k) = \sqrt{P_{\rm tracer}(k)/P_{\rm matter}(k)}$', fontsize=12)
    ax3.set_title('Bias Comparison', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0.020, 0.200)
    ax3.set_xscale('log')
    ax3.set_ylim(0.8, 2.0)
    
    # Add text with some statistics
    ax1.text(0.02, 0.98, f'Total k-modes: {len(k_gal):,}\nk range: [{k_gal.min():.3f}, {k_gal.max():.3f}] h/Mpc', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    output_dir = './galaxy_pk_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/galaxy_kPk_bias_step310.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/galaxy_kPk_bias_step310.pdf', bbox_inches='tight')
    
    print(f"Plot saved to: {output_file}")
    print(f"PDF version saved to: {output_dir}/galaxy_kPk_bias_step310.pdf")
    
    # Print some statistics
    print(f"\nPower spectrum statistics:")
    print(f"Galaxy k range: [{k_gal.min():.4f}, {k_gal.max():.4f}] h/Mpc")
    print(f"Halo k range: [{k_halo.min():.4f}, {k_halo.max():.4f}] h/Mpc") 
    print(f"Matter k range: [{k_matter.min():.4f}, {k_matter.max():.4f}] h/Mpc")
    print(f"Galaxy P(k) range (original): [{P_gal.min():.2e}, {P_gal.max():.2e}] (Mpc/h)³")
    print(f"Galaxy P(k) range (corrected): [{P_gal_corrected.min():.2e}, {P_gal_corrected.max():.2e}] (Mpc/h)³")
    print(f"Halo P(k) range: [{P_halo.min():.2e}, {P_halo.max():.2e}] (Mpc/h)³")
    print(f"Matter P(k) range: [{P_matter.min():.2e}, {P_matter.max():.2e}] (Mpc/h)³")
    print(f"Shot noise: {shot_noise:.2e} (Mpc/h)³")
    print(f"Galaxy k×P(k) range (corrected): [{kPk_gal.min():.2e}, {kPk_gal.max():.2e}] (Mpc/h)²")
    print(f"Total number of k-modes: {len(k_gal)}")
    print(f"Galaxy number density: {N_galaxies/volume:.2e} (h/Mpc)⁻³")
    print(f"\nBias statistics:")
    # Calculate bias statistics for k=0.02-0.1 range
    mask_gal_stats = mask_gal & (k_gal >= 0.02) & (k_gal <= 0.1)
    mask_halo_stats = mask_halo & (k_gal >= 0.02) & (k_gal <= 0.1)
    print(f"Galaxy bias (k=0.02-0.1): mean={bias_galaxy[mask_gal_stats].mean():.2f}, std={bias_galaxy[mask_gal_stats].std():.2f}")
    print(f"Halo bias (k=0.02-0.1): mean={bias_halo[mask_halo_stats].mean():.2f}, std={bias_halo[mask_halo_stats].std():.2f}")
    
    # Close the plot to free memory
    plt.close()

if __name__ == "__main__":
    plot_kPk() 