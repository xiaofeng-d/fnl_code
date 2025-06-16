import numpy as np
import numpy
import pygio
import numpy as np
import numpy
import scipy
from numpy import sqrt, sin, cos, pi, diff
import scipy.special as special
import scipy.integrate as integrate
import re
from scipy.special import erfc
import matplotlib.pyplot as plt
import random
import os 

from scipy.special import erf


from scipy.integrate import quad
num_seeds = 1
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000

def read_halofile(haloproperties_path, massbin_min=1.0e11, massbin_max=1.024e15):
    """
    Read halo properties from a GenericIO file and filter halos by mass range.
    """
    # Read the complete GenericIO file directly
    data = pygio.read_genericio(haloproperties_path)

    # Display available keys
    print("\nAvailable keys in halo file:")
    for key in sorted(data.keys()):
        print(f"- {key}")
    print()  # Add blank line for readability
    
    # Convert data to numpy arrays if not already
    for key in data:
        if not isinstance(data[key], np.ndarray):
            data[key] = np.array(data[key])
        print(f"{key}: {len(data[key])} entries, shape: {data[key].shape}")
    
    print(f'Only considering halo mass in range {massbin_min:.1e} to {massbin_max:.1e}')
    halo_mass = data['sod_halo_mass']

    # Filter by mass range and positive halo count
    indices = (
        (halo_mass > massbin_min) &
        (halo_mass < massbin_max) &
        (data['sod_halo_count'] > 0)
    )

    print(f'Total number of halos within mass bin: {indices.sum()}')
    print(f'Original number of halos: {len(halo_mass)}')

    # Apply filter to all data fields
    filtered_data = {key: values[indices] for key, values in data.items()}

    return filtered_data

# Define Cosmological Parameters
z = 1.0  # simulation redshift - 修改为z=1.0
omega_m = 0.28309 # matter density parameter
omega_lambda = 1- omega_m # dark energy density parameter

# Define HOD parameters
# f_ci = 1.0       # Normalization constant (example value)
# M_cut = 1e13     # Characteristic halo mass (example value)
# sigma =  0.9646      # Transition width (example value)
# M_1 = 1e14       # Halo mass scale for satellites (example value)
# alpha = 1.01616      # Power-law index (example value)
k = 0.5          # Offset constant (example value)

# ---------- ELG best-fit HOD : Rocher+25 (Table 6, modified profile) ----------
# central–galaxy HOD
Ac       = 0.107 #0.0204         # adjusted to reach target ng ~ 1.9e-4 (h/Mpc)^-3
log10_Mc = 11.64
sigmaM   = 0.30
gamma    = 5.47           # controls high‑mass tail asymmetry

# satellite HOD
As       = 2.47 #0.47         # adjusted to reach target ng ~ 1.9e-4 (h/Mpc)^-3
log10_M0 = 11.20          # cut‑off mass for satellites
alpha    = 0.81
M1_fix   = 1.0e13         # fixed in the fit – leave as 10^13 M☉/h

# modified NFW satellite positioning
f_exp       = 0.58        # fraction drawn from exponential tail
tau_exp     = 6.14        # slope parameter of the exponential
lambda_NFW  = 0.67        # stretches the NFW scale radius (rs → rs/λ)
# ---------------------------------------------------------------------------
Mc  = 10**log10_Mc
M0  = 10**log10_M0


# # Define the HOD functions
# def N_cen(M): # M is an array of halo masses
#     return (f_ci / 2) * erfc((np.log10(M_cut / M)) / (np.sqrt(2) * sigma))


# def N_sat(M):
#     n_cen = N_cen(M)
#     return ((M - k * M_cut) / M_1)**alpha * n_cen


# ---------- new HOD functions ----------
def N_cen(M):
    """
    mHMQ central occupation – eq. (3.4) of Rocher+25.
    M is an array [M_sun/h].
    """
    x = np.log10(M)
    gauss = (Ac / np.sqrt(2*np.pi*sigmaM**2)) * \
            np.exp(-(x - np.log10(Mc))**2 / (2*sigmaM**2))
    return gauss * (1.0 + erf(gamma * (x - np.log10(Mc)) /
                              (np.sqrt(2)*sigmaM)))

def N_sat(M):
    """
    Satellite occupation – eq. (3.5) with strict central–satellite conformity.
    """
    out = np.zeros_like(M)
    mask = M > M0
    out[mask] = As * ((M[mask] - M0)/M1_fix)**alpha
    return out  * N_cen(M)   # apply strict conformity
# ---------------------------------------

def nfw_density(r, rho_s, R_s):
    return rho_s / ((r/R_s) * (1 + r/R_s)**2)

# Define M200 for given rho_s and c
def M200(rho_s=1, r_s=1, c=5):
    return 4 * np.pi * rho_s * r_s**3 * (np.log(1+c) - c/(1+c))

# Define the NFW profile
def rho_NFW(r, rho_s=1, r_s=1):
    return rho_s / (r/r_s * (1 + r/r_s)**2)

# Define the integrated mass profile
def M_NFW(r, rho_s=1, r_s=1):
    integrand = lambda r_prime: 4 * np.pi * rho_s * r_prime**2 / (r_prime/r_s * (1 + r_prime/r_s)**2)
    mass, _ = quad(integrand, 0, r)
    return mass

# Define the normalized integrated mass profile
def normalized_M_NFW(r, rho_s=1, r_s=1, c=5, M200_val=1):
    # M200_val = M200(rho_s, r_s, c)
    return M_NFW(r, rho_s, r_s) / M200_val





# Given M200 and c, determine rho_s
def determine_rho_s(M200=1, c=5, r_s=1):
    R200 = c * r_s
    # Rearrange the M200 formula to solve for rho_s
    return M200 / (4 * np.pi * r_s**3 * (np.log(1+c) - c/(1+c)))

# run this function for the first time only, and generate CDF curves and r arrays
def precompute_CDF_M_over_M200(cmin, cmax, step=0.05):
    num_of_points = 1000 # points in CDF curve
    r_values_extended = np.linspace(0.01, 1, num_of_points)
    c_values = np.arange(cmin, cmax, step)
    # Fixed R200 for given M200
    R200_fixed = 1

    #critical_density = 1
    critical_density = 2.78e11*(omega_m*(1+z)**3 + omega_lambda)

    M200 = critical_density * 200 * 4/3 * np.pi * (R200_fixed**3)
    CDF = np.zeros((c_values.size, num_of_points))
    r_arr = np.zeros((c_values.size, num_of_points))

    plt.figure(dpi=300)
    # Plot for each c value
    for i, c in enumerate(c_values):
        r_s = R200_fixed / c  # r_s varies with c, for fixed R200
        rho_s = determine_rho_s(M200, c, r_s)
        M_values_normalized = [normalized_M_NFW(r, rho_s=rho_s, r_s=r_s, c=c, M200_val = M200) for r in r_values_extended if r <= R200_fixed]
        r_valid = [r for r in r_values_extended if r <= R200_fixed]  # Only consider radii up to R200
        
        CDF[i] = np.array(M_values_normalized)
        r_arr[i] = np.array(r_valid)

        plt.plot(r_valid, M_values_normalized, label=f'c={c}')
    np.save('/home/ac.xdong/hacc-bispec/galaxies/CDF_'+str(cmin)+'_'+str(cmax)+'.npy',CDF)
    np.save('/home/ac.xdong/hacc-bispec/galaxies/r_arr_'+str(cmin)+'_'+str(cmax)+'.npy',r_arr)
    np.save('/home/ac.xdong/hacc-bispec/galaxies/c_values_'+str(cmin)+'_'+str(cmax)+'.npy',c_values)
    print('precomputing finished for cmin, cmax = ',cmin, ',',cmax)
    plt.savefig('/home/ac.xdong/hacc-bispec/galaxies/crazyplot.png')
    plt.close()
    return 

c_values = np.load('/home/ac.xdong/hacc-bispec/galaxies/c_values_0_6.npy')
CDF = np.load('/home/ac.xdong/hacc-bispec/galaxies/CDF_0_6.npy')
r_arr = np.load('/home/ac.xdong/hacc-bispec/galaxies/r_arr_0_6.npy')

plt.figure(dpi=300)

# c_values store the pre-computed concentration values. concentration is the specific halo concentration in question

def lookup_CDF_curve(c_values, CDF, r_arr, concentration, size) -> float:  # can generate array

    # find closest c value
    diff = np.abs(c_values-concentration)
    cindx = np.argmin(diff)
    


    # generate random number and find the cloest value in CDF
    randnums = np.random.rand(size)
    # print(randnum)
    # print(CDF[cindx,:])
    diffs = np.abs(CDF[cindx, :] - randnums[:, np.newaxis])

    # rindx = np.argmin(np.abs(CDF[cindx,:]-randnum))
    rindxs = np.argmin(diffs, axis=1)
    # print(CDF[cindx,:]-randnum)
    # print(rindx)
    r_values = r_arr[cindx, rindxs]
    # plt.plot(r_arr[cindx,:],CDF[cindx,:], label='c = '+str(concentration))
    # # plt.axvline(x=r_arr[cindx,rindx],linestyle='dotted',alpha=0.6)
    # # plt.axhline(y=randnum,linestyle='dashed',color='orange',alpha=0.6)
    # plt.plot(r_arr[cindx,rindx],randnum, marker='+',markersize=10)
    # plt.close()

    return r_values #r_arr[cindx,rindxs]

def check3():
    for _ in range(5):
        print('a possible radius for concentration', _+1 , 'is: ', lookup_CDF_curve(c_values, CDF, r_arr, _+1))
    plt.xlabel('Radius (r)')
    plt.ylabel('Mass/M200')
    plt.title('Normalized Cumulative Mass of NFW Profile for different c values')
    plt.legend()
    plt.savefig('/home/dongx/hacc-bispec/galaxies/CDFcurve-lookup.png')




# precompute_CDF_M_over_M200(0,6)

def check2():
    # the r_values to ensure it covers up to R200 for the maximum c value
    r_values_extended = np.linspace(0.01, 25, 1000)  # extended to cover the range up to R200 for maximum c

    # Generate values for plotting
    r_values = np.linspace(0.001, 25, 500)  # start from 0.01 to avoid division by zero
    M_values = [M_NFW(r) for r in r_values]

    # Generate values for plotting
    c_values = [1, 2, 5, 10]
    colors = ['blue', 'red', 'green', 'purple']


    # Given R200, and assume a c, we compute Rs;
    # Use R200 and critical density, compute M200;
    # from M200 and Rs, compute rho_0;
    # integral and get the curve

    plt.figure(figsize=(10,6))
    # Fixed R200 for given M200
    R200_fixed = 1

    #critical_density = 1
    critical_density = 2.78e11*(omega_m*(1+z)**3 + omega_lambda)

    M200 = critical_density * 200 * 4/3 * np.pi * (R200_fixed**3)
    # Plot for each c value
    for i, c in enumerate(c_values):
        r_s = R200_fixed / c  # r_s varies with c, for fixed R200
        rho_s = determine_rho_s(M200, c, r_s)
        M_values_normalized = [normalized_M_NFW(r, rho_s=rho_s, r_s=r_s, c=c, M200_val = M200) for r in r_values_extended if r <= R200_fixed]
        r_valid = [r for r in r_values_extended if r <= R200_fixed]  # Only consider radii up to R200
        plt.plot(r_valid, M_values_normalized, label=f'c={c}', color=colors[i])

    plt.xlabel('Radius (r)')
    plt.ylabel('Mass/M200')
    plt.title('Normalized Cumulative Mass of NFW Profile for different c values')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/ac.xdong/hacc-bispec/galaxies/cumulativeM.png',dpi=300)


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sample_positions_from_modified_profile(halo_concentration, halo_radius, x_center=0, y_center=0, z_center=0, size_satellite=1, rng=np.random.default_rng()):
    """
    Draw *size_satellite* 3-D positions for satellites in *one* halo using modified profile.
    
    Parameters
    ----------
    halo_concentration : float
        Halo concentration c_200c.
    halo_radius : float
        Halo R_200c   [h^-1 Mpc].
    x_center, y_center, z_center : float
        Halo center coordinates (default 0).
    size_satellite : int
        Number of satellites to draw.
    rng : np.random.Generator
        Random-number generator instance.
    Returns
    -------
    x, y, z : 1-D arrays of length *size_satellite*  [h^-1 Mpc absolute coordinates]
    """

    # ---------- how many go into each component ----------------------------
    n_exp = int(np.round(f_exp * size_satellite))
    n_nfw = size_satellite - n_exp

    # ---------- exponential tail  r ∝ exp(−r / (τ rs))  -------------------
    rs = halo_radius / halo_concentration
    u  = rng.random(n_exp)
    r_exp = -tau_exp * rs * np.log(1.0 - u)        # inverse-CDF
    r_exp = np.clip(r_exp, 0.0, halo_radius)       # keep inside R_vir

    # ---------- stretched NFW component  ----------------------------------
    #   same routine you already have but with   c' = conc / λ
    if n_nfw > 0:
        x_nfw, y_nfw, z_nfw = sample_positions_from_nfw(halo_concentration / lambda_NFW, halo_radius, 
                                                        x_center=0, y_center=0, z_center=0, 
                                                        size_satellite=n_nfw, rng=rng)
        # Convert back to radial distances for concatenation
        r_nfw = np.sqrt(x_nfw**2 + y_nfw**2 + z_nfw**2)
    else:
        r_nfw = np.array([])

    # ---------- concatenate & convert to Cartesian -------------------------
    r = np.concatenate([r_exp, r_nfw])
    theta = np.arccos(1.0 - 2.0 * rng.random(size_satellite))
    phi   = 2.0 * np.pi * rng.random(size_satellite)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Add halo center coordinates
    x += x_center
    y += y_center
    z += z_center
    
    return x, y, z
# ---------------------------------------------------------------------------



def sample_positions_from_nfw(halo_concentration, halo_radius, x_center=0, y_center=0, z_center=0, size_satellite=1, rng=np.random.default_rng()):   
    # c_values, CDF, r_arr are pre-computed from 
    radii = lookup_CDF_curve(c_values,CDF,r_arr, halo_concentration, size_satellite) * halo_radius
    if len(radii)>0:
        assert(radii.min() >= 0.0 and radii.max() <= halo_radius)

    # Generate random angles for each radius - 可以选择使用传入的 rng 或继续使用 np.random
    phi = 2 * np.pi * np.random.rand(size_satellite)
    theta = np.arccos(2 * np.random.rand(size_satellite) - 1)
    
    # Convert to Cartesian coordinates
    x, y, z = spherical_to_cartesian(radii, theta, phi)
    
    return x + x_center, y + y_center, z + z_center


def gen_centerhalos(data, savepath, boxsize=2000):

    # Define HOD parameters
    M_min = 1e12  # Example value
    M_1 = 1e13   # Example value
    alpha = 1.0  # Example value

    # Extract halo properties from data
    halo_mass = data["sod_halo_mass"] #[::3]
    halo_x = data["sod_halo_center_x"] #[::3]
    halo_y = data["sod_halo_center_y"] #[::3]
    halo_z = data["sod_halo_center_z"] #[::3]
    halo_radius = data["sod_halo_radius"] #[::3]
    print('halo_radius', halo_radius)
    halo_concentration = data["sod_halo_cdelta"] #[::3]
    # print('halo_radius', halo_radius)

    halo_fofhalocenter = data["fof_halo_center_x"]
    print('fof halo center x: ', halo_fofhalocenter)

    # Populate halos with galaxies, determine
    Ncen_values = N_cen(halo_mass)
    Nsat_values = N_sat(halo_mass)
    print('min mass: {:.2e}'.format(min(halo_mass)),'max mass: {:.2E}'.format(max(halo_mass)))

    print('min Ncen_values: {:.2e}'.format(min(Ncen_values)),'max Ncen_values: {:.2E}'.format(max(Ncen_values)))

    # plt.plot(halo_mass, Ncen_values,label='Central',color='blue',alpha=0.8)
    # plt.plot(halo_mass, Nsat_values,label='Satellites',color='orange',alpha=0.8)
    

    # Determine actual number of central and satellite galaxies
    random_numbers_cen = np.random.rand(len(halo_mass))
    # generate central galaxy by a random number
    Ncen_actual = (random_numbers_cen < Ncen_values).astype(int)
    # generate satellite number w/ Poisson
    Nsat_actual = np.random.poisson(Nsat_values) 

    print('number of central halos and satellite halos:')
    print(Ncen_actual)
    print(Nsat_actual)

    # ============ 添加详细统计信息 ============
    print("\n" + "="*60)
    print("HOD GALAXY GENERATION STATISTICS")
    print("="*60)
    
    # 总暗晕数量
    total_halos = len(halo_mass)
    print(f"总输入暗晕数量: {total_halos:,}")
    
    # 中心星系统计
    total_central_galaxies = np.sum(Ncen_actual)
    halos_with_central = np.sum(Ncen_actual > 0)
    central_fraction = halos_with_central / total_halos * 100
    print(f"\n中心星系统计:")
    print(f"  - 中心星系总数: {total_central_galaxies:,}")
    print(f"  - 包含中心星系的暗晕数: {halos_with_central:,}")
    print(f"  - 中心星系占据率: {central_fraction:.2f}%")
    
    # 卫星星系统计
    total_satellite_galaxies = np.sum(Nsat_actual)
    halos_with_satellites = np.sum(Nsat_actual > 0)
    satellite_fraction = halos_with_satellites / total_halos * 100
    if halos_with_satellites > 0:
        avg_satellites_per_halo = total_satellite_galaxies / halos_with_satellites
    else:
        avg_satellites_per_halo = 0
    print(f"\n卫星星系统计:")
    print(f"  - 卫星星系总数: {total_satellite_galaxies:,}")
    print(f"  - 包含卫星星系的暗晕数: {halos_with_satellites:,}")
    print(f"  - 卫星星系占据率: {satellite_fraction:.2f}%")
    print(f"  - 每个有卫星的暗晕平均卫星数: {avg_satellites_per_halo:.2f}")
    
    # 总星系统计
    total_galaxies = total_central_galaxies + total_satellite_galaxies
    central_galaxy_fraction = total_central_galaxies / total_galaxies * 100 if total_galaxies > 0 else 0
    satellite_galaxy_fraction = total_satellite_galaxies / total_galaxies * 100 if total_galaxies > 0 else 0
    
    print(f"\n总星系统计:")
    print(f"  - 星系总数: {total_galaxies:,}")
    print(f"  - 中心星系占比: {central_galaxy_fraction:.2f}%")
    print(f"  - 卫星星系占比: {satellite_galaxy_fraction:.2f}%")
    
    # 产生星系的暗晕统计
    halos_with_any_galaxy = np.sum((Ncen_actual > 0) | (Nsat_actual > 0))
    galaxy_hosting_fraction = halos_with_any_galaxy / total_halos * 100
    print(f"\n星系宿主暗晕统计:")
    print(f"  - 产生星系的暗晕总数: {halos_with_any_galaxy:,}")
    print(f"  - 星系宿主暗晕占比: {galaxy_hosting_fraction:.2f}%")
    
    # 按质量分bin显示统计
    mass_bins = [1e11, 1e12, 1e13, 1e14, 1e15]
    print(f"\n按质量范围统计:")
    for i in range(len(mass_bins)-1):
        mask = (halo_mass >= mass_bins[i]) & (halo_mass < mass_bins[i+1])
        if np.sum(mask) > 0:
            halos_in_bin = np.sum(mask)
            cen_in_bin = np.sum(Ncen_actual[mask])
            sat_in_bin = np.sum(Nsat_actual[mask])
            print(f"  - 质量范围 [{mass_bins[i]:.0e}, {mass_bins[i+1]:.0e}): ")
            print(f"    暗晕数: {halos_in_bin:,}, 中心星系: {cen_in_bin:,}, 卫星星系: {sat_in_bin:,}")
    
    print("="*60)
    # ============ 统计信息结束 ============

    # ------------------------------------------------------------------
    def check_hod(savepath="hod_comparison_rocher.png"):
        """
        Bin the simulated 〈N〉 vs M, overlay the analytic Rocher+25 curves,
        and write the figure to *savepath*. Also save arrays for later use.
        """
        # ---------- binning the mock ------------
        nbins = 18
        mass_bins = np.logspace(np.log10(halo_mass.min()),
                                np.log10(halo_mass.max()),
                                nbins + 1)
        bin_idx = np.digitize(halo_mass, mass_bins) - 1

        avg_ncen, avg_nsat, err_ncen, err_nsat, bin_cent = [], [], [], [], []
        for i in range(nbins):
            sel = bin_idx == i
            if sel.sum() == 0:        # skip empty bins
                continue
            bin_cent.append(np.sqrt(mass_bins[i] * mass_bins[i+1]))
            avg_ncen.append(Ncen_actual[sel].mean())
            avg_nsat.append(Nsat_actual[sel].mean())
            err_ncen.append(Ncen_actual[sel].std(ddof=1) / np.sqrt(sel.sum()))
            err_nsat.append(Nsat_actual[sel].std(ddof=1) / np.sqrt(sel.sum()))
        bin_cent = np.array(bin_cent)

        # ---------- analytic curves -------------
        M_grid = np.logspace(11, 14.8, 300)
        N_cen_analytic = N_cen(M_grid)
        N_sat_analytic = N_sat(M_grid)
        
        # ---------- save data for later use -----
        data_file = os.path.splitext(savepath)[0] + "_arrays.npz"
        np.savez(data_file,
                 bin_centers=bin_cent,
                 avg_ncen=np.array(avg_ncen),
                 avg_nsat=np.array(avg_nsat),
                 err_ncen=np.array(err_ncen),
                 err_nsat=np.array(err_nsat),
                 M_grid=M_grid,
                 N_cen_analytic=N_cen_analytic,
                 N_sat_analytic=N_sat_analytic,
                 raw_halo_mass=halo_mass,
                 raw_Ncen_actual=Ncen_actual,
                 raw_Nsat_actual=Nsat_actual)
        print(f"HOD data arrays saved to {data_file}")

        # ---------- plotting -------------------
        plt.figure(figsize=(6.0,4.8), dpi=300)
        plt.loglog(M_grid, N_cen_analytic, 'k-',  lw=1.6,
                label="Rocher + 25 analytic $N_\\mathrm{cen}$")
        plt.loglog(M_grid, N_sat_analytic, 'k--', lw=1.6,
                label="Rocher + 25 analytic $N_\\mathrm{sat}$")

        plt.errorbar(bin_cent, avg_ncen, yerr=err_ncen,
                    fmt='o', ms=4, color='cornflowerblue',
                    label="simulation ⟨$N_\\mathrm{cen}$⟩")
        plt.errorbar(bin_cent, avg_nsat, yerr=err_nsat,
                    fmt='s', ms=4, color='orange',
                    label="simulation ⟨$N_\\mathrm{sat}$⟩")

        plt.xlabel(r"$M_{200c}\;[h^{-1}M_\odot]$")
        plt.ylabel(r"$\langle N\,|\,M\rangle$")
        plt.xlim(1e11, 5e14)
        plt.ylim(3e-3, 5e1)
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(savepath)
        print(f"HOD comparison plot written to {savepath}")
        plt.close()
    # ------------------------------------------------------------------

    def check():
        # Define bin edges (for statistics plot)
        bin_edges = np.linspace(halo_mass.min(), halo_mass.max(), 15)
        # Find the bin each halo_mass belongs to
        bin_indices = np.digitize(halo_mass, bin_edges)

        # Compute average and standard error for each bin
        average_Ncen = []
        standard_error = []
        average_Nsat = []
        standard_error_sat = []
        for i in range(1, len(bin_edges)):
            mask = bin_indices == i
            bin_Ncen = Ncen_actual[mask]
            bin_Nsat = Nsat_actual[mask]

            avg = bin_Ncen.mean()
            avg_sat = bin_Nsat.mean()

            # print('bin_Ncen size',bin_Ncen.size)
            std_err = bin_Ncen.std() / np.sqrt(bin_Ncen.size)
            std_err_sat = bin_Nsat.std() / np.sqrt(bin_Nsat.size)

            average_Ncen.append(avg)
            average_Nsat.append(avg_sat)

            standard_error.append(std_err)
            standard_error_sat.append(std_err_sat)

        average_Ncen = np.array(average_Ncen)
        average_Nsat = np.array(average_Nsat)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.plot(bin_centers, average_Ncen, label='Average Ncen',color='cornflowerblue')
        plt.fill_between(bin_centers, average_Ncen - standard_error, average_Ncen + standard_error, color='cornflowerblue', alpha=0.5)

        plt.plot(bin_centers, average_Nsat, label='Average Nsat',color='orange')
        plt.fill_between(bin_centers, average_Nsat - standard_error_sat, average_Nsat + standard_error_sat, color='orange', alpha=0.5)


        # plt.plot(bin_centers, Ncen_actual,label='Central',color='blue',alpha=0.4,marker='.',linestyle='None')
        # plt.plot(halo_mass, Nsat_actual,label='Satellites',color='orange',alpha=0.4,marker='.',linestyle='None')
        


    # Lists to store generated satellite positions
    sat_x = []
    sat_y = []
    sat_z = []
    sat_c = []
    sat_mass = []
    galaxy_data = {}
    galaxy_data["fof_halo_center_x"] = []
    galaxy_data["fof_halo_center_y"] = []
    galaxy_data["fof_halo_center_z"] = []
    galaxy_data["fof_halo_tag"] = []
    galaxy_data["sod_halo_cdelta"] = []
    galaxy_data["sod_halo_mass"] = []

    # Loop through halos and generate satellite positions
    c_values = np.load('/home/ac.xdong/hacc-bispec/galaxies/c_values_0_6.npy') #pre-computed concentration values
    CDF = np.load('/home/ac.xdong/hacc-bispec/galaxies/CDF_0_6.npy') # pre-computed CDF curves
    r_arr = np.load('/home/ac.xdong/hacc-bispec/galaxies/r_arr_0_6.npy') # pre-computed radius from 0-1
    # for plotting and check
    cent_sat_dict = {}
    displacements = []
    displacements_x = []
    displacements_y = []
    displacements_z = []

    for i in range(len(halo_mass)):
        if Ncen_actual[i] == 1:  # If central halo is populated
            # print(galaxy_data["fof_halo_center_x"])
            galaxy_data["fof_halo_center_x"].extend([halo_x[i]])
            galaxy_data["fof_halo_center_y"].extend([halo_y[i]])
            galaxy_data["fof_halo_center_z"].extend([halo_z[i]])
            galaxy_data["fof_halo_tag"].extend([2*i]) ## 'even number' for central galaxy
            galaxy_data["sod_halo_cdelta"].extend([halo_concentration[i]])
            galaxy_data["sod_halo_mass"].extend([halo_mass[i]])
        # add satellite x, y, z from central halo position
        if Ncen_actual[i] == 1:
            # print('actual satellite number', Nsat_actual[i])
            # rho_s = 1.0
            # R_s = 1.0

            # NFW profile, selected modified profile only to benchmark with another plot
            # x, y, z = sample_positions_from_nfw(halo_concentration[i], halo_radius[i], size_satellite=Nsat_actual[i])
            x, y, z = sample_positions_from_modified_profile(
                halo_concentration[i], halo_radius[i],
                x_center=halo_x[i], y_center=halo_y[i], z_center=halo_z[i],
                size_satellite=Nsat_actual[i])

            # Calculate displacements from halo center
            for num, item in enumerate(x):
                dx = x[num] - halo_x[i]
                dy = y[num] - halo_y[i] 
                dz = z[num] - halo_z[i]
                displacements.append(np.sqrt(dx**2 + dy**2 + dz**2))
            # x, y, z are already in absolute coordinates from sample_positions_from_modified_profile
            # print('test x: length',len(x),x)
            sat_x.extend(x)
            sat_y.extend(y)
            sat_z.extend(z)
            sat_c.extend([halo_concentration[i]]*Nsat_actual[i])
            sat_mass.extend([halo_mass[i]]*Nsat_actual[i])

            ## write down
            tuplets = (list(zip(x, y, z))) 
            cent_sat_dict[(halo_x[i],halo_y[i],halo_z[i])] = tuplets
    
    ## plot central-satellites to visually check 
    fig = plt.figure(dpi=300,figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    items = list(cent_sat_dict.items())
    sample_size = max(1, len(items) // 20)  # Ensure at least 1 sample if len(items) < 100
    sampled_items = random.sample(items, sample_size)

    # Iterate through the dictionary
    for central, satellites in sampled_items:
        # Central galaxy
        if satellites == []:
            continue
        # print('satellites',satellites)
        ax.scatter(*central, color='blue', marker='o',alpha=0.5, s=1)  # Central galaxy as a blue dot

        # Satellites
        satellites = np.array(satellites)  # Convert list of tuples to numpy array
        ax.scatter(satellites[:,0], satellites[:,1], satellites[:,2], color='red', marker='^',alpha=0.2, s=0.1)  # Satellites as red triangles

        # Draw lines from central galaxy to each satellite
        for sat in satellites:
            ax.plot([central[0], sat[0]], [central[1], sat[1]], [central[2], sat[2]], color='gray',linewidth=3)

    # Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.savefig('./central-satellite-check.png')
    print('plot done!! ')
    plt.close()
    ## Create the histogram
    plt.figure(dpi=300)
    plt.hist(displacements, bins='auto', color='blue', alpha=0.7)

    # Add labels and title
    plt.xlabel('Displacement')
    plt.ylabel('Frequency')
    plt.title('Histogram of Displacements')
    plt.savefig('./central-satellite-check-histo.png')
    
    check_hod()
    
    # input("Press Enter to continue...")




    # Add these to the data
    l = len(sat_x)
    # l = len(galaxy_data["fof_halo_center_x"])

    galaxy_data["fof_halo_tag"].extend([2*i+1 for i in range(l)] ) ## 'odd numbers' for satellite galaxy
    galaxy_data["fof_halo_center_x"].extend(sat_x)
    galaxy_data["fof_halo_center_y"].extend(sat_y)
    galaxy_data["fof_halo_center_z"].extend(sat_z)
    galaxy_data["sod_halo_cdelta"].extend(sat_c)
    galaxy_data["sod_halo_mass"].extend(sat_mass)

    # convert list to numpy array
    galaxy_data["fof_halo_center_x"] = np.array(galaxy_data["fof_halo_center_x"])
    galaxy_data["fof_halo_center_y"] = np.array(galaxy_data["fof_halo_center_y"])
    galaxy_data["fof_halo_center_z"] = np.array(galaxy_data["fof_halo_center_z"])
    galaxy_data["fof_halo_tag"] = np.array(galaxy_data["fof_halo_tag"])
    galaxy_data["sod_halo_cdelta"] = np.array(galaxy_data["sod_halo_cdelta"])
    galaxy_data["sod_halo_mass"] = np.array(galaxy_data["sod_halo_mass"])

    # fake data generation
    galaxy_data["fof_halo_mass"] = np.ones(len(galaxy_data["fof_halo_center_x"]))
    galaxy_data["sod_halo_com_x"] = np.zeros(len(galaxy_data["fof_halo_center_x"]))
    galaxy_data["sod_halo_com_y"] = np.zeros(len(galaxy_data["fof_halo_center_x"]))
    galaxy_data["sod_halo_com_z"] = np.zeros(len(galaxy_data["fof_halo_center_x"]))
    

    # Assuming "dw" is defined and you have proper tags and masses for these satellites
    # You should handle the addition of these values
    
    # dw = '/home/dongx/hacc-bispec/galaxies/'
    for key in galaxy_data:
        print(key)
        print(galaxy_data[key].shape)
        if key != 'fof_halo_tag':
            galaxy_data[key] = np.float32(galaxy_data[key])
        else:
            galaxy_data[key] = np.int64(galaxy_data[key])
    # pygio.write_genericio(savepath, galaxy_data, [boxsize, boxsize, boxsize], [0, 0, 0])

    # 保存galaxy数据
    print(f'Saving galaxy data to: {savepath}')
    pygio.write_genericio(savepath, galaxy_data, [boxsize, boxsize, boxsize], [0, 0, 0])
    
    # return Ncen_actual.sum(), Nsat_actual.sum(), galaxy_data
    return Ncen_actual.sum(), len(sat_x), galaxy_data


def extract_mass_values(mass_bins):
    mass_min = []
    mass_max = []
    
    for bin_str in mass_bins:
        # Extract values using regex
        values = re.findall(r"(\d+\.\d+e\d+)", bin_str)
        
        # Append values to respective lists
        if len(values) == 2:
            mass_min.append(float(values[0]))
            mass_max.append(float(values[1]))

            
    return mass_min, mass_max

## actual driver code

if __name__ == "__main__":
    # 设置参数
    boxsize = 2000  # 模拟盒子大小 (Mpc/h)
    simulations = ['gauss', 's8l', 's8h', 'fnl1', 'fnl10']  # 要处理的模拟类型
    snapshots = [310]  # 修改为处理310 step (z=1.0)
    base_path = "/scratch/cpac/emberson/SPHEREx/L2000"
    
    # 创建输出目录
    output_dir = "./galaxies"
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有模拟和快照
    for sim_name in simulations:
        print(f"\n\n{'='*80}")
        print(f"Processing simulation: {sim_name}")
        print(f"{'='*80}")
        
        # 构建模拟目录路径
        if sim_name == 'gauss':
            sim_dir = f"{base_path}/output_l2000n4096_gauss_tpm_seed0"
        elif sim_name in ['s8l', 's8h']:
            sim_dir = f"{base_path}/output_l2000n4096_gauss_tpm_seed0_{sim_name}"
        elif sim_name in ['fnl1', 'fnl10']:
            sim_dir = f"{base_path}/output_l2000n4096_{sim_name}_tpm_seed0"
        else:
            continue  # 跳过未知的模拟类型
        
        for snap in snapshots:
            print(f"\nProcessing snapshot: {snap}")
            
            # 构建输入路径
            halos_dir = f"{sim_dir}/HALOS-b0168"
            input_file = f"{halos_dir}/m000-{snap}.haloproperties"
            
            # 构建输出路径（存储在当前目录的galaxies子目录下）
            output_file = f"{output_dir}/{sim_name}_m000-{snap}.galaxies.haloproperties"
            
            # 读取halo数据
            print(f"Reading halo data from: {input_file}")
            try:
                data = read_halofile(input_file)
                
                # 生成中心galaxy和卫星galaxy
                n_cen, n_sat, galaxy_data = gen_centerhalos(data, output_file, boxsize=boxsize)
                
                print(f"Processed snapshot {snap} for {sim_name}:")
                print(f"- Input file: {input_file}")
                print(f"- Output file: {output_file}")
                print(f"- Central galaxies: {n_cen}")
                print(f"- Satellite galaxies: {n_sat}")
                print(f"- Total galaxies: {n_cen + n_sat}")
                print(f"- Success: Galaxy data saved to {output_file}")
                
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")
                continue

