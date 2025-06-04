#!/usr/bin/env python3
"""
Compute linear bias b1 and growth bias bphi for galaxies, and compare power spectra of matter, galaxies, and halos.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import sys
import scipy.interpolate as interp
import matplotlib as mpl

# ────────────────── Configuration ──────────────────
SNAP = "624"
KMAX = 0.05                     # Linear scale upper limit (h/Mpc)

# Halo mass range (M⊙/h)
HALO_MASS_MIN = 1.0e11
HALO_MASS_MAX = 3.16e11

# Add second halo mass range (M⊙/h)
HALO_MASS_MIN_2 = 3.16e11
HALO_MASS_MAX_2 = 1.00e12

# Data directories
GALAXY_PK_DIR = "/scratch/cpac/emberson/SPHEREx/L2000/power_hod/new"
BASE_GAUSS = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/")
OUTPUT_DIR = "./results_galaxy_bias"

# Simulation parameters
SIGMA8_G = 0.834   # Baseline σ₈
SIGMA8_H = 0.844   # High σ₈
SIGMA8_L = 0.824   # Low σ₈

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

# Use system available fonts, avoid Times New Roman font issues
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['figure.dpi'] = 300

# ───────────────────────────────────────────────────

def mass_str(x: float) -> str:
    """Convert mass value to string format"""
    return f"{x:.2e}".replace("+", "")   # 1.00e+13 → 1.00e13

def load_power(path, col_k=0, col_p=1):
    """
    Read power spectrum file, return k and P(k)
    """
    print(f"Reading power spectrum: {path}")
    try:
        if not os.path.exists(path):
            print(f"Error: File does not exist {path}")
            return None, None
            
        data = np.loadtxt(path, comments='#')
        k = data[:, col_k]
        P = data[:, col_p]
        return k, P
    except Exception as e:
        print(f"Error reading file {path}: {str(e)}")
        return None, None

def interpolate_power(k_target, k_source, P_source):
    """
    Interpolate power spectrum from source k values to target k values
    
    Parameters:
        k_target: Target k value array
        k_source: Source k value array
        P_source: Source power spectrum array
    
    Returns:
        P_target: Interpolated power spectrum array
    """
    # Check k value range
    k_min = max(k_target.min(), k_source.min())
    k_max = min(k_target.max(), k_source.max())
    
    # Create masks
    mask_target = (k_target >= k_min) & (k_target <= k_max)
    mask_source = (k_source >= k_min) & (k_source <= k_max)
    
    # Interpolate in log space
    log_interp = interp.interp1d(
        np.log10(k_source[mask_source]), 
        np.log10(P_source[mask_source]), 
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    # Apply interpolation
    P_target = np.zeros_like(k_target)
    P_target[mask_target] = 10**log_interp(np.log10(k_target[mask_target]))
    
    return P_target

def compute_b1(k, Pg, Pm):
    """
    Compute linear bias b1(k) = sqrt(Pg/Pm)
    """
    return np.sqrt(Pg / Pm)

def compute_b1_mean(k, b1_k):
    """
    Compute mean b1 in linear regime
    """
    mask = k <= KMAX
    return np.mean(b1_k[mask])

def compute_bphi(Pg_h, Pg_l):
    """
    Compute growth bias bphi = d ln(Pg) / d ln(sigma8)
    """
    # Calculate ln(Pg_h/Pg_l) / ln(sigma8_h/sigma8_l)
    return np.log(Pg_h/Pg_l) / np.log(SIGMA8_H/SIGMA8_L)

def check_power_spectrum_units(k, P, name):
    """Check basic properties of power spectrum"""
    print(f"\nChecking {name} power spectrum:")
    print(f"k range: {k.min():.6e} to {k.max():.6e} h/Mpc")
    print(f"P range: {P.min():.6e} to {P.max():.6e} (Mpc/h)^3")
    print(f"Number of k points: {len(k)}")
    
    # Check P(k) behavior on large scales
    large_scale_mask = k < 0.01
    if np.any(large_scale_mask):
        print(f"Large scale P(k) (k < 0.01): {P[large_scale_mask].mean():.6e}")
    
    # Check P(k) behavior on small scales
    small_scale_mask = k > 1.0
    if np.any(small_scale_mask):
        print(f"Small scale P(k) (k > 1.0): {P[small_scale_mask].mean():.6e}")
    
    # Plot power spectrum
    plt.figure(figsize=(10, 6))
    plt.loglog(k, P, 'o-', lw=1, ms=3)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P(k) [(Mpc/h)^3]')
    plt.title(f'{name} Power Spectrum')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    
    # Use safe filename (replace special characters)
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('²', '2').replace('³', '3').replace('¹', '1').replace('⊙', 'sun')
    plt.savefig(f"{OUTPUT_DIR}/{safe_name}_check.png")
    plt.close()

def plot_power_spectra(k, Pm, Pg, Ph, Ph_extra, b1_k, b1_mean, output_file):
    """
    Plot power spectra and bias curves
    
    Parameters:
        k: k value array
        Pm: Matter power spectrum
        Pg: Galaxy power spectrum
        Ph: Halo power spectrum (10^13 range)
        Ph_extra: Additional halo power spectrum (10^12 range)
        b1_k: Linear bias b1(k)
        b1_mean: Mean linear bias
        output_file: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Upper panel: Power spectra
    ax1.loglog(k, Pm, 'k-', lw=2, label='Matter P(k)')
    ax1.loglog(k, Pg, 'r-', lw=2, label='Galaxy P(k)')
    
    if Ph is not None:
        ax1.loglog(k, Ph, 'b--', lw=2, label='Halo P(k) (10¹³ M⊙/h)')
    
    if Ph_extra is not None:
        ax1.loglog(k, Ph_extra, 'g-.', lw=2, label='Halo P(k) (10¹² M⊙/h)')
    
    # Mark linear regime
    ax1.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set axes
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.set_title('Power Spectrum and Galaxy Bias', fontsize=16)
    ax1.legend(fontsize=12, frameon=True)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # Lower panel: Bias
    ax2.semilogx(k, b1_k, 'r-', lw=2, label=f'b₁(k), ⟨b₁⟩ = {b1_mean:.3f}')
    
    # Mark mean bias
    ax2.axhline(y=b1_mean, color='k', ls='--', label=f'⟨b₁⟩ = {b1_mean:.3f}')
    
    # Mark linear regime
    ax2.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set y-axis range to around 2 for better bias observation
    ax2.set_ylim(0, 3)
    
    # Set axes
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('b₁(k)', fontsize=14)
    ax2.legend(fontsize=12, frameon=True)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close()

def plot_all_power_spectra(k_m, Pm, Ph, Ph_extra, galaxy_results, output_file):
    """
    Plot all power spectra for comparison on one figure (no interpolation, show original curves)
    
    Parameters:
        k_m: Matter power spectrum k value array
        Pm: Matter power spectrum
        Ph: Halo power spectrum (10^13 range)
        Ph_extra: Additional halo power spectrum (not used)
        galaxy_results: Dictionary containing galaxy power spectra from various simulations
        output_file: Output file path
    """
    # Define colors and marker styles
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot power spectrum comparison
    plt.figure(figsize=(12, 8))
    
    # Plot matter power spectrum
    plt.loglog(k_m, Pm, 'k-', lw=2.5, label='Matter P(k)')
    
    # Plot halo power spectrum (only 10^13 range)
    if Ph is not None:
        # Re-read original halo power spectrum data
        halo_pk_file = f"{BASE_GAUSS}/POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_1.00e11_mh_3.16e11.pk"
        k_h_orig, Ph_orig = load_power(halo_pk_file)
        if Ph_orig is not None:
            plt.loglog(k_h_orig, Ph_orig, 'b--', lw=2.5, label='Halo P(k) (1e11-3.16e11 M_sun/h)')
    
    # Plot galaxy power spectra from various simulations (no interpolation, use original data)
    for i, (sim_name, pk_file) in enumerate({
        'gauss': f"{GALAXY_PK_DIR}/pk_hod_gauss.dat.pk",
        's8h': f"{GALAXY_PK_DIR}/pk_hod_s8h.dat.pk",
        's8l': f"{GALAXY_PK_DIR}/pk_hod_s8l.dat.pk",
        'fnl1': f"{GALAXY_PK_DIR}/pk_hod_fnl1.dat.pk",
        'fnl10': f"{GALAXY_PK_DIR}/pk_hod_fnl10.dat.pk"
    }.items()):
        # Re-read original galaxy power spectrum data
        k_g_orig, Pg_orig = load_power(pk_file)
        if Pg_orig is not None:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Skip every few points to avoid overly dense plots
            skip = max(1, len(k_g_orig) // 30)
            
            plt.loglog(k_g_orig[::skip], Pg_orig[::skip], 
                      color=color, marker=marker, markersize=4, 
                      linestyle='-', lw=1.5, 
                      label=f'Galaxy P(k) - {sim_name}')
    
    # Mark linear regime
    plt.axvline(x=KMAX, color='gray', ls='--', alpha=0.7, label=f'Linear limit (k={KMAX})')
    
    # Set axes
    plt.xlabel('k [h/Mpc]', fontsize=14)
    plt.ylabel('P(k) [(Mpc/h)^3]', fontsize=14)
    plt.title('Power Spectrum Comparison (Original Data)', fontsize=16)
    
    # Add legend
    plt.legend(fontsize=10, frameon=True, loc='best')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Power spectrum comparison plot saved to: {output_file}")
    plt.close()

def get_galaxy_number_density():
    """
    Read galaxy count directly from galaxy files and calculate number density
    """
    # Galaxy file paths
    galaxy_files = {
        'gauss': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/gauss_m000-624.galaxies.haloproperties",
        's8h': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/s8h_m000-624.galaxies.haloproperties", 
        's8l': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/s8l_m000-624.galaxies.haloproperties",
        'fnl1': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/fnl1_m000-624.galaxies.haloproperties",
        'fnl10': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/fnl10_m000-624.galaxies.haloproperties"
    }
    
    # Simulation box volume (Mpc/h)^3
    boxsize = 2000.0  # Mpc/h
    volume = boxsize**3
    
    # Calculate number density (h/Mpc)^3
    galaxy_number_densities = {}
    
    for sim_name, galaxy_file in galaxy_files.items():
        try:
            import pygio
            data = pygio.read_genericio(galaxy_file)
            
            # Get total galaxy count directly (no mass filtering needed)
            # Assume each row represents a galaxy, get count through any field length
            if 'x' in data:
                galaxy_count = len(data['x'])
            elif 'pos_x' in data:
                galaxy_count = len(data['pos_x'])
            else:
                # If uncertain about field names, take the length of the first field
                first_key = list(data.keys())[0]
                galaxy_count = len(data[first_key])
            
            galaxy_density = galaxy_count / volume
            galaxy_number_densities[sim_name] = galaxy_density
            
            print(f"{sim_name} galaxy count: {galaxy_count}")
            print(f"{sim_name} galaxy number density: {galaxy_density:.6e} (h/Mpc)^3")
            
        except Exception as e:
            print(f"Cannot read galaxy file {galaxy_file}: {e}")
            # Use previous estimates as fallback
            fallback_counts = {
                'gauss': 7233385,
                's8h': 7229491, 
                's8l': 7268868,
                'fnl1': 7238274,
                'fnl10': 7235730
            }
            galaxy_count = fallback_counts.get(sim_name, 7000000)  # Default value
            galaxy_density = galaxy_count / volume
            galaxy_number_densities[sim_name] = galaxy_density
            print(f"Using fallback value - {sim_name} galaxy count: {galaxy_count}")
            print(f"Using fallback value - {sim_name} galaxy number density: {galaxy_density:.6e} (h/Mpc)^3")
    
    return galaxy_number_densities

def get_halo_number_density():
    """
    Calculate number density of halos in different mass ranges
    """
    # Read halo file to get counts
    halo_file = f"{BASE_GAUSS}/HALOS-b0168/m000-{SNAP}.haloproperties"
    
    try:
        import pygio
        data = pygio.read_genericio(halo_file)
        halo_mass = data["sod_halo_mass"]
        
        # Calculate halo count in 1e11-3.16e11 mass range
        mask_11_1 = (halo_mass >= HALO_MASS_MIN) & (halo_mass <= HALO_MASS_MAX)
        # Calculate halo count in 3.16e11-1e12 mass range
        mask_11_2 = (halo_mass >= HALO_MASS_MIN_2) & (halo_mass <= HALO_MASS_MAX_2)
        
        count_11_1 = np.sum(mask_11_1)
        count_11_2 = np.sum(mask_11_2)
        
        # Simulation box volume
        boxsize = 2000.0  # Mpc/h
        volume = boxsize**3
        
        density_11_1 = count_11_1 / volume
        density_11_2 = count_11_2 / volume
        
        print(f"Halo count (1e11-3.16e11 range): {count_11_1}")
        print(f"Halo number density (1e11-3.16e11 range): {density_11_1:.6e} (h/Mpc)^3")
        print(f"Halo count (3.16e11-1e12 range): {count_11_2}")
        print(f"Halo number density (3.16e11-1e12 range): {density_11_2:.6e} (h/Mpc)^3")
        
        return {'halo_11_1': density_11_1, 'halo_11_2': density_11_2}
        
    except Exception as e:
        print(f"Cannot read halo file: {e}")
        # Use estimated values
        estimated_densities = {
            'halo_11_1': 1e-3,  # 1e11-3.16e11 range
            'halo_11_2': 5e-4   # 3.16e11-1e12 range (fewer in number)
        }
        return estimated_densities

def subtract_shot_noise(k, P, number_density):
    """
    Subtract shot noise from power spectrum
    
    Parameters:
        k: k value array
        P: Power spectrum array
        number_density: Number density (h/Mpc)^3
    
    Returns:
        P_corrected: Power spectrum after shot noise subtraction
    """
    shot_noise = 1.0 / number_density
    P_corrected = P - shot_noise
    
    print(f"Shot noise: {shot_noise:.6e}")
    print(f"Power spectrum range (original): {P.min():.6e} - {P.max():.6e}")
    print(f"Power spectrum range (corrected): {P_corrected.min():.6e} - {P_corrected.max():.6e}")
    
    return P_corrected

def plot_power_spectra_shot_noise_corrected(k, Pm, Pg, Ph, Ph_extra, b1_k, b1_mean, 
                                           galaxy_density, halo_densities, sim_name, output_file):
    """
    Plot power spectra and bias curves after shot noise subtraction
    """
    # Subtract shot noise
    Pg_corrected = subtract_shot_noise(k, Pg, galaxy_density)
    
    if Ph is not None:
        Ph_corrected = subtract_shot_noise(k, Ph, halo_densities['halo_11_1'])  # Use 1e11 range density
    else:
        Ph_corrected = None
        
    if Ph_extra is not None:
        Ph_extra_corrected = subtract_shot_noise(k, Ph_extra, halo_densities['halo_11_2'])
    else:
        Ph_extra_corrected = None
    
    # Recalculate bias (using corrected galaxy power spectrum)
    b1_k_corrected = compute_b1(k, Pg_corrected, Pm)
    b1_mean_corrected = compute_b1_mean(k, b1_k_corrected)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Upper panel: Power spectra (original and corrected)
    ax1.loglog(k, Pm, 'k-', lw=2, label='Matter P(k)')
    ax1.loglog(k, Pg, 'r--', lw=1.5, alpha=0.7, label='Galaxy P(k) (original)')
    ax1.loglog(k, Pg_corrected, 'r-', lw=2, label='Galaxy P(k) (shot noise subtracted)')
    
    if Ph is not None and Ph_corrected is not None:
        ax1.loglog(k, Ph, 'b--', lw=1.5, alpha=0.7, label='Halo P(k) 1e11-3.16e11 (original)')
        ax1.loglog(k, Ph_corrected, 'b-', lw=2, label='Halo P(k) 1e11-3.16e11 (shot noise subtracted)')
    
    if Ph_extra is not None and Ph_extra_corrected is not None:
        ax1.loglog(k, Ph_extra, 'g--', lw=1.5, alpha=0.7, label='Halo P(k) 3.16e11-1e12 (original)')
        ax1.loglog(k, Ph_extra_corrected, 'g-', lw=2, label='Halo P(k) 3.16e11-1e12 (shot noise subtracted)')
    
    # Mark linear regime
    ax1.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set axes
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.set_title(f'Power Spectrum with Shot Noise Correction - {sim_name}', fontsize=16)
    ax1.legend(fontsize=10, frameon=True)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # Lower panel: Bias comparison
    ax2.semilogx(k, b1_k, 'r--', lw=1.5, alpha=0.7, label=f'b₁(k) original, ⟨b₁⟩ = {b1_mean:.3f}')
    ax2.semilogx(k, b1_k_corrected, 'r-', lw=2, label=f'b₁(k) corrected, ⟨b₁⟩ = {b1_mean_corrected:.3f}')
    
    # Mark mean bias
    ax2.axhline(y=b1_mean, color='k', ls='--', alpha=0.7, label=f'⟨b₁⟩ original = {b1_mean:.3f}')
    ax2.axhline(y=b1_mean_corrected, color='k', ls='-', label=f'⟨b₁⟩ corrected = {b1_mean_corrected:.3f}')
    
    # Mark linear regime
    ax2.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set y-axis range
    ax2.set_ylim(0, 3)
    
    # Set axes
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('b₁(k)', fontsize=14)
    ax2.legend(fontsize=10, frameon=True)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Shot noise corrected plot saved to: {output_file}")
    plt.close()

def calculate_galaxy_bias():
    """Calculate galaxy bias and compare power spectra"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get galaxy and halo number densities
    galaxy_densities = get_galaxy_number_density()
    halo_densities = get_halo_number_density()
    
    # Read matter power spectrum
    matter_pk_file = f"{BASE_GAUSS}/POWER/m000.pk.{SNAP}"
    k_m, Pm = load_power(matter_pk_file)
    if k_m is None:
        print("Error: Cannot read matter power spectrum")
        return
    
    # Check matter power spectrum
    check_power_spectrum_units(k_m, Pm, "Matter")
    
    # Read first halo power spectrum (1e11-3.16e11 range)
    halo_pk_file_1 = f"{BASE_GAUSS}/POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_1.00e11_mh_3.16e11.pk"
    k_h1, Ph1 = load_power(halo_pk_file_1)
    
    # Read second halo power spectrum (3.16e11-1.00e12 range)
    halo_pk_file_2 = f"{BASE_GAUSS}/POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_3.16e11_mh_1.00e12.pk"
    k_h2, Ph2 = load_power(halo_pk_file_2)
    
    # Check halo power spectra
    if Ph1 is not None:
        check_power_spectrum_units(k_h1, Ph1, "Halo (1e11-3.16e11 M⊙/h)")
        # If k values don't match, interpolate
        if not np.array_equal(k_m, k_h1):
            print("First halo power spectrum k values differ from matter power spectrum, interpolating")
            Ph1 = interpolate_power(k_m, k_h1, Ph1)
    else:
        print("Warning: Cannot read first halo power spectrum")
    
    if Ph2 is not None:
        check_power_spectrum_units(k_h2, Ph2, "Halo (3.16e11-1e12 M⊙/h)")
        # If k values don't match, interpolate
        if not np.array_equal(k_m, k_h2):
            print("Second halo power spectrum k values differ from matter power spectrum, interpolating")
            Ph2 = interpolate_power(k_m, k_h2, Ph2)
    else:
        print("Warning: Cannot read second halo power spectrum")
    
    # Read galaxy power spectra - using new file paths
    galaxy_pk_files = {
        'gauss': f"{GALAXY_PK_DIR}/pk_hod_gauss.dat.pk",
        's8h': f"{GALAXY_PK_DIR}/pk_hod_s8h.dat.pk",
        's8l': f"{GALAXY_PK_DIR}/pk_hod_s8l.dat.pk",
        'fnl1': f"{GALAXY_PK_DIR}/pk_hod_fnl1.dat.pk",
        'fnl10': f"{GALAXY_PK_DIR}/pk_hod_fnl10.dat.pk"
    }
    
    # Store results
    results = {}
    
    # Process each simulation
    for sim_name, pk_file in galaxy_pk_files.items():
        print(f"\nProcessing simulation: {sim_name}")
        
        # Read galaxy power spectrum
        k_g, Pg = load_power(pk_file)
        if k_g is None:
            print(f"Skipping simulation {sim_name}: Cannot read galaxy power spectrum")
            continue
        
        # Check galaxy power spectrum
        check_power_spectrum_units(k_g, Pg, f"Galaxy ({sim_name})")
        
        # Ensure k values match, interpolate if they don't
        if not np.array_equal(k_m, k_g):
            print(f"Warning: {sim_name} k values differ from matter power spectrum, interpolating")
            Pg = interpolate_power(k_m, k_g, Pg)
            k_g = k_m  # Use matter power spectrum k values as unified standard
        
        # Calculate b1(k)
        b1_k = compute_b1(k_g, Pg, Pm)
        
        # Calculate mean b1 in linear regime
        b1_mean = compute_b1_mean(k_g, b1_k)
        
        # Store results
        results[sim_name] = {
            'k': k_g,
            'Pg': Pg,
            'b1_k': b1_k,
            'b1_mean': b1_mean
        }
        
        # Plot original power spectra and bias curves (including two halo power spectra)
        output_file = f"{OUTPUT_DIR}/power_bias_{sim_name}.png"
        plot_power_spectra_dual_halos(k_g, Pm, Pg, Ph1, Ph2, b1_k, b1_mean, output_file)
        
        # Plot shot noise corrected figures
        if sim_name in galaxy_densities and 'halo_11_1' in halo_densities and 'halo_11_2' in halo_densities:
            output_file_corrected = f"{OUTPUT_DIR}/power_bias_{sim_name}_shot_noise_corrected.png"
            plot_power_spectra_shot_noise_corrected_dual(
                k_g, Pm, Pg, Ph1, Ph2, b1_k, b1_mean,
                galaxy_densities[sim_name], halo_densities, sim_name, output_file_corrected
            )
        
        print(f"{sim_name} mean linear bias b1 = {b1_mean:.3f}")
    
    # Plot all power spectra comparison
    all_pk_output_file = f"{OUTPUT_DIR}/all_power_spectra_comparison.png"
    plot_all_power_spectra_dual(k_m, Pm, Ph1, Ph2, results, all_pk_output_file)

def plot_power_spectra_dual_halos(k, Pm, Pg, Ph1, Ph2, b1_k, b1_mean, output_file):
    """
    Plot power spectra and bias curves (including two halo power spectra)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Upper panel: Power spectra
    ax1.loglog(k, Pm, 'k-', lw=2, label='Matter P(k)')
    ax1.loglog(k, Pg, 'r-', lw=2, label='Galaxy P(k)')
    
    if Ph1 is not None:
        ax1.loglog(k, Ph1, 'b-', lw=2, label='Halo P(k) (1e11-3.16e11 M⊙/h)')
    
    if Ph2 is not None:
        ax1.loglog(k, Ph2, 'g-', lw=2, label='Halo P(k) (3.16e11-1e12 M⊙/h)')
    
    # Mark linear regime
    ax1.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set axes
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.set_title('Power Spectrum Comparison (Dual Halo Mass Ranges)', fontsize=16)
    ax1.legend(fontsize=10, frameon=True)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # Lower panel: Bias
    ax2.semilogx(k, b1_k, 'r-', lw=2, label=f'b₁(k), ⟨b₁⟩ = {b1_mean:.3f}')
    
    # Mark mean bias
    ax2.axhline(y=b1_mean, color='k', ls='--', alpha=0.7, label=f'⟨b₁⟩ = {b1_mean:.3f}')
    
    # Mark linear regime
    ax2.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set y-axis range
    ax2.set_ylim(0, 3)
    
    # Set axes
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('b₁(k)', fontsize=14)
    ax2.legend(fontsize=10, frameon=True)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Dual halo mass range plot saved to: {output_file}")
    plt.close()

def plot_power_spectra_shot_noise_corrected_dual(k, Pm, Pg, Ph1, Ph2, b1_k, b1_mean, 
                                                galaxy_density, halo_densities, sim_name, output_file):
    """
    Plot power spectra and bias curves after shot noise subtraction (dual halo mass ranges)
    """
    # Subtract shot noise
    Pg_corrected = subtract_shot_noise(k, Pg, galaxy_density)
    
    Ph1_corrected = None
    Ph2_corrected = None
    
    if Ph1 is not None:
        Ph1_corrected = subtract_shot_noise(k, Ph1, halo_densities['halo_11_1'])
    
    if Ph2 is not None:
        Ph2_corrected = subtract_shot_noise(k, Ph2, halo_densities['halo_11_2'])
    
    # Recalculate bias
    b1_k_corrected = compute_b1(k, Pg_corrected, Pm)
    b1_mean_corrected = compute_b1_mean(k, b1_k_corrected)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Upper panel: Power spectra (original and corrected)
    ax1.loglog(k, Pm, 'k-', lw=2, label='Matter P(k)')
    ax1.loglog(k, Pg, 'r--', lw=1.5, alpha=0.7, label='Galaxy P(k) (original)')
    ax1.loglog(k, Pg_corrected, 'r-', lw=2, label='Galaxy P(k) (shot noise subtracted)')
    
    if Ph1 is not None and Ph1_corrected is not None:
        ax1.loglog(k, Ph1, 'b--', lw=1.5, alpha=0.7, label='Halo P(k) 1e11-3.16e11 (original)')
        ax1.loglog(k, Ph1_corrected, 'b-', lw=2, label='Halo P(k) 1e11-3.16e11 (shot noise subtracted)')
    
    if Ph2 is not None and Ph2_corrected is not None:
        ax1.loglog(k, Ph2, 'g--', lw=1.5, alpha=0.7, label='Halo P(k) 3.16e11-1e12 (original)')
        ax1.loglog(k, Ph2_corrected, 'g-', lw=2, label='Halo P(k) 3.16e11-1e12 (shot noise subtracted)')
    
    # Mark linear regime
    ax1.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set axes
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.set_title(f'Power Spectrum with Shot Noise Correction (Dual Halo) - {sim_name}', fontsize=16)
    ax1.legend(fontsize=9, frameon=True)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # Lower panel: Bias comparison
    ax2.semilogx(k, b1_k, 'r--', lw=1.5, alpha=0.7, label=f'b₁(k) original, ⟨b₁⟩ = {b1_mean:.3f}')
    ax2.semilogx(k, b1_k_corrected, 'r-', lw=2, label=f'b₁(k) corrected, ⟨b₁⟩ = {b1_mean_corrected:.3f}')
    
    # Mark mean bias
    ax2.axhline(y=b1_mean, color='k', ls='--', alpha=0.7, label=f'⟨b₁⟩ original = {b1_mean:.3f}')
    ax2.axhline(y=b1_mean_corrected, color='k', ls='-', label=f'⟨b₁⟩ corrected = {b1_mean_corrected:.3f}')
    
    # Mark linear regime
    ax2.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # Set y-axis range
    ax2.set_ylim(0, 3)
    
    # Set axes
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('b₁(k)', fontsize=14)
    ax2.legend(fontsize=10, frameon=True)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Dual halo shot noise corrected plot saved to: {output_file}")
    plt.close()

def plot_all_power_spectra_dual(k_m, Pm, Ph1, Ph2, galaxy_results, output_file):
    """
    Plot all power spectra for comparison on one figure (including dual halo mass ranges)
    """
    # Define colors and marker styles
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    plt.figure(figsize=(12, 8))
    
    # Plot matter power spectrum
    plt.loglog(k_m, Pm, 'k-', lw=3, label='Matter P(k)')
    
    # Plot first halo power spectrum
    if Ph1 is not None:
        halo_pk_file_1 = f"{BASE_GAUSS}/POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_1.00e11_mh_3.16e11.pk"
        k_h1_orig, Ph1_orig = load_power(halo_pk_file_1)
        if Ph1_orig is not None:
            plt.loglog(k_h1_orig, Ph1_orig, 'b-', lw=2.5, label='Halo P(k) (1e11-3.16e11 M_sun/h)')
    
    # Plot second halo power spectrum
    if Ph2 is not None:
        halo_pk_file_2 = f"{BASE_GAUSS}/POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_3.16e11_mh_1.00e12.pk"
        k_h2_orig, Ph2_orig = load_power(halo_pk_file_2)
        if Ph2_orig is not None:
            plt.loglog(k_h2_orig, Ph2_orig, 'g-', lw=2.5, label='Halo P(k) (3.16e11-1e12 M_sun/h)')
    
    # Plot galaxy power spectra
    for i, (sim_name, result) in enumerate(galaxy_results.items()):
        k_g = result['k']
        Pg = result['Pg']
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.loglog(k_g, Pg, color=color, marker=marker, markersize=4, 
                  linewidth=2, markevery=5, label=f'Galaxy P(k) ({sim_name})')
    
    # Set axes
    plt.xlabel('k [h/Mpc]', fontsize=14)
    plt.ylabel('P(k) [(Mpc/h)^3]', fontsize=14)
    plt.title('Power Spectrum Comparison (Dual Halo Mass Ranges)', fontsize=16)
    
    # Add legend
    plt.legend(fontsize=10, frameon=True, loc='best')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Dual halo power spectrum comparison plot saved to: {output_file}")
    plt.close()

def main():
    """Main function"""
    print("Computing galaxy bias and comparing power spectra")
    calculate_galaxy_bias()
    print("Processing completed")

if __name__ == "__main__":
    main() 