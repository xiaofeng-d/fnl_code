import numpy as np
import matplotlib.pyplot as plt

def print_bias_values(k, b, label, mass_range):
    # 选择一些典型的k值来打印
    k_points = [0.01, 0.1, 1.0]
    print(f"\nBias values for {label}, Mass range: {mass_range}")
    for k_target in k_points:
        idx = np.abs(k - k_target).argmin()
        print(f"k = {k[idx]:.3f} h/Mpc: b = {b[idx]:.3f}")

def load_power_spectrum(filename):
    """Load power spectrum data from file."""
    data = np.loadtxt(filename, skiprows=1)
    k = data[:, 0]      # Wavenumber (h/Mpc)
    P = data[:, 1]      # Monopole P0(k) (Mpc/h)^3
    err = data[:, 2]    # Error bars
    return k, P, err

def format_mass(m):
    """Format mass number without plus sign in scientific notation"""
    return f"{m:.2e}".replace('+', '')

def plot_power_spectra():
    # Set up the plots: 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Define consistent mass bins for both redshifts
    mass_bins = [
        (1.00e12, 3.16e12),  # ~10^12 M☉/h
        (1.00e13, 3.16e13),  # ~10^13 M☉/h
        (1.00e14, 3.16e14)   # ~10^14 M☉/h
    ]
    colors = ['red', 'blue', 'green']
    
    base_path = '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/POWER'
    
    # Plot for z=0 (step 624)
    matter_file = f'{base_path}/m000.pk.624'
    k_m, P_m, err_m = load_power_spectrum(matter_file)
    ax1.errorbar(k_m, P_m, yerr=err_m, label='Matter', color='black')
    
    # Load and plot halo power spectra and bias for different mass bins (step 624)
    for (ml, mh), color in zip(mass_bins, colors):
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.624.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, err_h = load_power_spectrum(halo_file)
        
        # Plot power spectrum
        ax1.errorbar(k_h, P_h, yerr=err_h, 
                    label=f'SOD Halos ({format_mass(ml)} - {format_mass(mh)} M⊙/h)', color=color)
        
        # Calculate and plot bias
        bias = np.sqrt(P_h / P_m)  # b(k) = sqrt(P_hh/P_mm)
        # Error propagation for bias
        bias_err = 0.5 * bias * np.sqrt((err_h/P_h)**2 + (err_m/P_m)**2)
        ax3.errorbar(k_h, bias, yerr=bias_err, 
                    label=f'Mass bin: {format_mass(ml)} - {format_mass(mh)}', color=color)
    
    ax1.set_title('z = 0 (step 624), b0168')
    ax3.set_title('Bias z = 0, b0168')
    
    # Plot for z=1 (step 310)
    matter_file = f'{base_path}/m000.pk.310'
    k_m, P_m, err_m = load_power_spectrum(matter_file)
    ax2.errorbar(k_m, P_m, yerr=err_m, label='Matter', color='black')
    
    # Load and plot halo power spectra and bias for different mass bins (step 310)
    for (ml, mh), color in zip(mass_bins, colors):
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.310.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, err_h = load_power_spectrum(halo_file)
        
        # Plot power spectrum
        ax2.errorbar(k_h, P_h, yerr=err_h, 
                    label=f'SOD Halos ({format_mass(ml)} - {format_mass(mh)} M⊙/h)', color=color)
        
        # Calculate and plot bias
        bias = np.sqrt(P_h / P_m)
        bias_err = 0.5 * bias * np.sqrt((err_h/P_h)**2 + (err_m/P_m)**2)
        ax4.errorbar(k_h, bias, yerr=bias_err, 
                    label=f'Mass bin: {format_mass(ml)} - {format_mass(mh)}', color=color)
    
    ax2.set_title('z = 1 (step 310), b0168')
    ax4.set_title('Bias z = 1, b0168')
    
    # Customize power spectrum plots
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)³]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-2, 2)
        ax.grid(True)
        ax.legend()
    
    # Customize bias plots
    for ax in [ax3, ax4]:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$b(k) = \sqrt{P_{hh}/P_{mm}}$')
        ax.set_xscale('log')
        ax.grid(True)
        ax.set_xlim(1e-2, 2)
        ax.legend()
    
    # Set specific y-axis limits for each bias plot
    ax3.set_ylim(0, 10)    # z=0 bias plot
    ax4.set_ylim(0, 50)   # z=1 bias plot
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('power_spectra_comparison.png', dpi=300)
    plt.close()

def plot_bias_comparison():
    # Set up the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define consistent mass bins for both redshifts
    mass_bins = [
        (1.00e12, 3.16e12),  # ~10^12 M☉/h
        (1.00e13, 3.16e13),  # ~10^13 M☉/h
        (1.00e14, 3.16e14)   # ~10^14 M☉/h
    ]
    colors = ['red', 'blue', 'green']
    
    base_path = '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/POWER'
    
    # Plot for z=0 (step 624)
    matter_file = f'{base_path}/m000.pk.624'
    k_m, P_m, _ = load_power_spectrum(matter_file)
    
    # Load and plot biases for different mass bins (step 624)
    for (ml, mh), color in zip(mass_bins, colors):
        # Auto power spectrum
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.624.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, _ = load_power_spectrum(halo_file)
        
        # Cross power spectrum
        cross_file = f'{base_path}/HALOS-b0168/m000.hm.sod.624.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_c, P_c, _ = load_power_spectrum(cross_file)
        
        # Calculate both biases
        b_auto = np.sqrt(P_h / P_m)  # b(k) = sqrt(P_hh/P_mm)
        b_cross = P_c / P_m          # b(k) = P_hm/P_mm
        
        # Plot both biases
        label = f'Mass bin: {format_mass(ml)} - {format_mass(mh)}'
        ax1.plot(k_h, b_auto, label=f'{label} (auto)', color=color, ls='--')
        ax1.plot(k_c, b_cross, label=f'{label} (cross)', color=color)
        
        # 计算bias后添加打印
        print_bias_values(k_h, b_auto, "Auto bias z=0", f"{format_mass(ml)} - {format_mass(mh)}")
        print_bias_values(k_c, b_cross, "Cross bias z=0", f"{format_mass(ml)} - {format_mass(mh)}")
    
    ax1.set_title('Bias comparison z = 0 (step 624), b0168')
    
    # Plot for z=1 (step 310)
    matter_file = f'{base_path}/m000.pk.310'
    k_m, P_m, _ = load_power_spectrum(matter_file)
    
    # Load and plot biases for different mass bins (step 310)
    for (ml, mh), color in zip(mass_bins, colors):
        # Auto power spectrum
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.310.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, _ = load_power_spectrum(halo_file)
        
        # Cross power spectrum
        cross_file = f'{base_path}/HALOS-b0168/m000.hm.sod.310.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_c, P_c, _ = load_power_spectrum(cross_file)
        
        # Calculate both biases
        b_auto = np.sqrt(P_h / P_m)
        b_cross = P_c / P_m
        
        # Plot both biases
        label = f'Mass bin: {format_mass(ml)} - {format_mass(mh)}'
        ax2.plot(k_h, b_auto, label=f'{label} (auto)', color=color, ls='--')
        ax2.plot(k_c, b_cross, label=f'{label} (cross)', color=color)
        
        # 计算bias后添加打印
        print_bias_values(k_h, b_auto, "Auto bias z=1", f"{format_mass(ml)} - {format_mass(mh)}")
        print_bias_values(k_c, b_cross, "Cross bias z=1", f"{format_mass(ml)} - {format_mass(mh)}")
    
    ax2.set_title('Bias comparison z = 1 (step 310), b0168')
    
    # Customize plots
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$b(k)$')
        ax.set_xscale('log')
        ax.grid(True)
        ax.set_xlim(1e-2, 2)
        ax.legend()
    
    # Set y-axis limits
    ax1.set_ylim(0, 10)    # z=0 bias plot
    ax2.set_ylim(0, 50)    # z=1 bias plot
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('bias_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_power_spectra()
    plot_bias_comparison()