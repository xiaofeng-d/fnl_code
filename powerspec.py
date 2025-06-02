import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy 
import scipy
# Add the directory to Python's search path
sys.path.insert(0, "/home/ac.xdong/hacc-bispec")


import ComputeGrowth
from ComputeGrowth import GrowthFactor, GrowthFactor_unnorm,  d_gf_cdmrnu, H_cdmrnu, Omega_nu_massive_a

############ FUNCTIONS ##################
def GrowthFactor(z, params):
    """
    Compute the cosmological scale-invariant growth factor at redshift z.
    """

    z_primoridal = 100000.

    x1 = 1./(1. + z_primoridal)
    x2 = 1./(1. + z)
    x  = numpy.array([x1, x2])
    y1 = numpy.array([x1, 0.])

    y2 = scipy.integrate.odeint(d_gf_cdmrnu, y1, x, args=(params,))[1]
    dplus = y2[0]
    ddot  = y2[1]

    x1 = 1./(1. + z_primoridal)
    x2 = 1.
    x  = numpy.array([x1, x2])
    y1 = numpy.array([x1, 0.])

    y2 = scipy.integrate.odeint(d_gf_cdmrnu, y1, x, args=(params,))[1]
    gf = dplus/y2[0]
    gd = ddot/y2[0]

    D = gf
    dD = gd

    return D, dD

def d_gf_cdmrnu(y, a, params):
    """
    Growth factor ODE to be integrated (see Adrian's code for more details).
    """

    dydx = 0.*y
    H = H_cdmrnu(params, a)
    dydx[0] = y[1]/a/H
    dydx[1] = -2.*y[1]/a + 1.5*params["omega_cb"]*y[0]/(H*a**4)

    return dydx

def H_cdmrnu(params, a):
    """
    Hubble factor as a function of a.
    """

    om_r  = (1. + params["f_nu_massless"]) * params["omega_r"]/a**4
    om_cb = params["omega_cb"]/a**3
    om_nm = Omega_nu_massive_a(params, a)
    om_lm = params["omega_l"]
    om_k  = params["omega_k"]/a**2

    return numpy.sqrt(om_r + om_cb + om_nm + om_lm + om_k)

def Omega_nu_massive_a(params, a):
    """
    Chosen to correspond to either its corresponding radiation term or matter term -- whichever is larger.
    """

    mat = params["omega_n"]/a**3
    rad = params["f_nu_massive"]*params["omega_r"]/a**4

    return (mat>=rad)*mat + (rad>mat)*rad
################## PARAMETERS #######################


params = { }
params["omega_c"]        = 0.26067
params["deut"]           = 0.02242
params["h"]              = 0.6766
params["t_cmb"]          = 2.726
params["omega_n"]        = 0.0
params["n_eff_massless"] = 3.04
params["n_eff_massive"]  = 0.0
params["omega_k"]        = 0.0

params["omega_b"]          = params["deut"]/params["h"]**2
params["omega_cb"]         = params["omega_c"] + params["omega_b"]
params["omega_r"]          = 2.471e-5/params["h"]**2 * (params["t_cmb"]/2.725)**4
params["f_nu_massless"]    = (7./8.) * (4./11.)**(4./3.) * params["n_eff_massless"]
params["f_nu_massive"]     = (7./8.) * (4./11.)**(4./3.) * params["n_eff_massive"]
params["omega_n_massless"] = params["f_nu_massless"] * params["omega_r"]
params["omega_l"]          = 1.0 - (params["omega_cb"] + params["omega_r"] + params["omega_n_massless"] + params["omega_n"] + params["omega_k"])


z = 0
D_0, dD_0 = GrowthFactor(z, params)

z = 1 
D_1, dD_1 = GrowthFactor(z, params)

z = 200
D_init, dD_init = GrowthFactor(z, params)

############################################
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
    
    # Load linear power spectrum from IC
    linear_file = f'{base_path}/m000.pk.ini'
    k_lin, P_lin, err_lin = load_power_spectrum(linear_file)
    
    # Plot for z=0 (step 624)
    matter_file = f'{base_path}/m000.pk.624'
    k_m, P_m, err_m = load_power_spectrum(matter_file)
    ax1.errorbar(k_m, P_m, yerr=err_m, label='Matter', color='black')
    # Scale IC by growth factor squared for z=0
    P_lin_scaled = P_lin * (D_0/D_init)**2
    ax1.plot(k_lin, P_lin_scaled, label='IC (scaled)', color='gray', ls='--')
    
    # Load and plot halo power spectra and bias for different mass bins (step 624)
    for (ml, mh), color in zip(mass_bins, colors):
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.624.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, err_h = load_power_spectrum(halo_file)
        
        # Plot power spectrum
        ax1.errorbar(k_h, P_h, yerr=err_h, 
                    label=f'SOD Halos ({format_mass(ml)} - {format_mass(mh)} M⊙/h)', color=color)
        
        # Calculate and plot both biases
        b_nl = np.sqrt(P_h / P_m)  # Non-linear bias
        b_lin = np.sqrt(P_h / P_lin_scaled)  # Linear bias using scaled IC
        
        ax3.plot(k_h, b_nl, label=f'Non-linear, {format_mass(ml)} - {format_mass(mh)}', color=color)
        ax3.plot(k_h, b_lin, label=f'IC (scaled), {format_mass(ml)} - {format_mass(mh)}', color=color, ls='--')
    
    ax1.set_title('z = 0 (step 624), b0168')
    ax3.set_title('Bias z = 0, b0168')
    
    # Plot for z=1 (step 310)
    matter_file = f'{base_path}/m000.pk.310'
    k_m, P_m, err_m = load_power_spectrum(matter_file)
    ax2.errorbar(k_m, P_m, yerr=err_m, label='Matter', color='black')
    # Scale IC by growth factor squared for z=1
    P_lin_scaled = P_lin * (D_1/D_init)**2
    ax2.plot(k_lin, P_lin_scaled, label='IC (scaled)', color='gray', ls='--')
    
    # Load and plot halo power spectra and bias for different mass bins (step 310)
    for (ml, mh), color in zip(mass_bins, colors):
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.310.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, err_h = load_power_spectrum(halo_file)
        
        # Plot power spectrum
        ax2.errorbar(k_h, P_h, yerr=err_h, 
                    label=f'SOD Halos ({format_mass(ml)} - {format_mass(mh)} M⊙/h)', color=color)
        
        # Calculate and plot both biases
        b_nl = np.sqrt(P_h / P_m)  # Non-linear bias
        b_lin = np.sqrt(P_h / P_lin_scaled)  # Linear bias using scaled IC
        
        ax4.plot(k_h, b_nl, label=f'Non-linear, {format_mass(ml)} - {format_mass(mh)}', color=color)
        ax4.plot(k_h, b_lin, label=f'IC (scaled), {format_mass(ml)} - {format_mass(mh)}', color=color, ls='--')
    
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
        ax.set_ylabel(r'$b(k)$')  # Simplified label since we show both definitions
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
    
    # Load linear power spectrum from IC
    linear_file = f'{base_path}/m000.pk.ini'
    k_lin, P_lin, _ = load_power_spectrum(linear_file)
    
    # Plot for z=0 (step 624)
    matter_file = f'{base_path}/m000.pk.624'
    k_m, P_m, _ = load_power_spectrum(matter_file)
    P_lin_scaled = P_lin * (D_0/D_init)**2  # Scale IC for z=0
    
    # Load and plot biases for different mass bins (step 624)
    for (ml, mh), color in zip(mass_bins, colors):
        # Auto power spectrum
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.624.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, _ = load_power_spectrum(halo_file)
        
        # Cross power spectrum
        cross_file = f'{base_path}/HALOS-b0168/m000.hm.sod.624.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_c, P_c, _ = load_power_spectrum(cross_file)
        
        # Calculate both auto and linear biases
        b_auto = np.sqrt(P_h / P_m)
        b_lin = np.sqrt(P_h / P_lin_scaled)
        b_cross = P_c / P_m
        
        # Plot all three biases
        label = f'Mass bin: {format_mass(ml)} - {format_mass(mh)}'
        ax1.plot(k_h, b_auto, label=f'{label} (b_auto)', color=color, ls='--')
        ax1.plot(k_h, b_lin, label=f'{label} (b_lin, scaled)', color=color, ls=':')
        ax1.plot(k_c, b_cross, label=f'{label} (b_cross)', color=color)
    
    ax1.set_title('Bias comparison z = 0 (step 624), b0168')
    
    # Plot for z=1 (step 310)
    matter_file = f'{base_path}/m000.pk.310'
    k_m, P_m, _ = load_power_spectrum(matter_file)
    P_lin_scaled = P_lin * (D_1/D_init)**2  # Scale IC for z=1
    
    # Load and plot biases for different mass bins (step 310)
    for (ml, mh), color in zip(mass_bins, colors):
        # Auto power spectrum
        halo_file = f'{base_path}/HALOS-b0168/m000.hh.sod.310.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_h, P_h, _ = load_power_spectrum(halo_file)
        
        # Cross power spectrum
        cross_file = f'{base_path}/HALOS-b0168/m000.hm.sod.310.ml_{format_mass(ml)}_mh_{format_mass(mh)}.pk'
        k_c, P_c, _ = load_power_spectrum(cross_file)
        
        # Calculate both auto and linear biases
        b_auto = np.sqrt(P_h / P_m)
        b_lin = np.sqrt(P_h / P_lin_scaled)
        b_cross = P_c / P_m
        
        # Plot all three biases
        label = f'Mass bin: {format_mass(ml)} - {format_mass(mh)}'
        ax2.plot(k_h, b_auto, label=f'{label} (b_auto)', color=color, ls='--')
        ax2.plot(k_h, b_lin, label=f'{label} (b_lin, scaled)', color=color, ls=':')
        ax2.plot(k_c, b_cross, label=f'{label} (b_cross)', color=color)
    
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