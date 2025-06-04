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

from scipy.integrate import quad
num_seeds = 1
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000

def read_halofile(haloproperties_path, massbin_min=1.0e11, massbin_max=1.024e15):
    """Read halo properties from multiple files and combine them."""
    # Initialize empty lists to store data
    all_data = {}
    
    # Read the first file to get the keys
    data_0 = pygio.read_genericio(f"{haloproperties_path}#0")
    print("\nAvailable keys in halo file:")
    for key in sorted(data_0.keys()):
        print(f"- {key}")
    print()  # Add blank line for readability
    
    for key in data_0.keys():
        all_data[key] = []
    
    # Read all 256 files and accumulate data
    for i in range(256):
        try:
            data = pygio.read_genericio(f"{haloproperties_path}#{i}")
            for key in data.keys():
                all_data[key].extend(data[key])
        except Exception as e:
            print(f"Warning: Could not read file #{i}: {e}")
            continue
    
    # Convert lists to numpy arrays
    for key in all_data.keys():
        all_data[key] = np.array(all_data[key])
    
    print('only considering halo mass of range {:.1e} to {:.1e}'.format(massbin_min, massbin_max))
    halo_mass = all_data['sod_halo_mass']
    
    # Filter by mass range
    halo_mass_pos = halo_mass[halo_mass>0]
    print('min halo mass: {:.2e}'.format(min(halo_mass_pos)))
    
    indices = (halo_mass > massbin_min) & (halo_mass < massbin_max) & \
             (all_data['sod_halo_count'] > 0) & (all_data['sod_halo_cdelta'] > 0)  #sod_halo_count_dm
    
    print('total number of halos within mass bin', indices.sum())
    print('original number of halos', len(halo_mass))
    
    filtered_data = {}
    for key, values in all_data.items():
        filtered_data[key] = values[indices]
    
    return filtered_data






# Define Cosmological Parameters
z = 0.0 #simulation redshift
omega_m = 0.28309 # matter density parameter
omega_lambda = 1- omega_m # dark energy density parameter

# Define HOD parameters
f_ci = 1.0       # Normalization constant (example value)
M_cut = 1e13     # Characteristic halo mass (example value)
sigma =  0.9646      # Transition width (example value)
M_1 = 1e14       # Halo mass scale for satellites (example value)
alpha = 1.01616      # Power-law index (example value)
k = 0.5          # Offset constant (example value)

# Define the HOD functions
def N_cen(M): # M is an array of halo masses
    return (f_ci / 2) * erfc((np.log10(M_cut / M)) / (np.sqrt(2) * sigma))


def N_sat(M):
    n_cen = N_cen(M)
    return ((M - k * M_cut) / M_1)**alpha * n_cen

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

def sample_positions_from_nfw(halo_concentration, halo_radius, x_center=0, y_center=0, z_center=0, size_satellite=1):   # x,y,z are center halo position, size refers to number of samples
# size 

    # c_values, CDF, r_arr are pre-computed from 
    radii = lookup_CDF_curve(c_values,CDF,r_arr, halo_concentration, size_satellite) * halo_radius
    if len(radii)>0:
        assert(radii.min() >= 0.0 and radii.max() <= halo_radius)
    # radii = sample_radius_from_nfw(rho_s, R_s, size)
    # print('halo radius is: ',halo_radius)

    # Generate random angles for each radius
    phi = 2 * np.pi * np.random.rand(size_satellite)
    theta = np.arccos(2 * np.random.rand(size_satellite) - 1)  # This ensures uniform sampling on a sphere
    
    # Convert to Cartesian coordinates
    x, y, z = spherical_to_cartesian(radii, theta, phi)
    
    return x + x_center, y + y_center, z + z_center


def gen_centerhalos(data, savepath, boxsize=1600):

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
            x, y, z = sample_positions_from_nfw(halo_concentration[i], halo_radius[i], size_satellite=Nsat_actual[i])
            for num, item in enumerate(x):
                displacements.append(np.sqrt(x[num]**2+y[num]**2+z[num]**2))
            # print(x,y,z)
            x += halo_x[i]
            y += halo_y[i]
            z += halo_z[i]
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
    
    
    input("Press Enter to continue...")




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
    pygio.write_genericio(savepath, galaxy_data, [boxsize, boxsize, boxsize], [0, 0, 0])   ### need to comment back 



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

def plot_halo_mass_function(data, output_dir='./', sim_name='gauss'):
    """Plot the halo mass function from halo properties."""
    halo_mass = data["sod_halo_mass"]
    
    # Create mass bins in log space
    mass_bins = np.logspace(np.log10(halo_mass.min()), np.log10(halo_mass.max()), 30)
    
    # Calculate the volume of the simulation box
    boxsize = 2000.0  # Mpc/h
    volume = boxsize**3  # (Mpc/h)^3
    
    # Calculate the halo mass function
    hist, bin_edges = np.histogram(halo_mass, bins=mass_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    dlogM = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
    
    # Convert to dn/dlogM
    hmf = hist / (volume * dlogM)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.loglog(bin_centers, hmf, 'o-', label=f'{sim_name}')
    
    # Add vertical line at 1e11 M☉/h
    plt.axvline(x=1e11, color='r', linestyle='--', label='1e11 M☉/h')
    
    plt.xlabel('Halo Mass [M☉/h]')
    plt.ylabel('dn/dlogM [(Mpc/h)⁻³]')
    plt.title(f'Halo Mass Function - {sim_name}')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(f'{output_dir}/halo_mass_function_{sim_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print some statistics
    print(f"\nHalo Mass Function Statistics for {sim_name}:")
    print(f"Total number of halos: {len(halo_mass)}")
    print(f"Mass range: {halo_mass.min():.2e} - {halo_mass.max():.2e} M☉/h")
    print(f"Number of halos > 1e11 M☉/h: {np.sum(halo_mass > 1e11)}")
    print(f"Number of halos 1e11-3.16e11 M☉/h: {np.sum((halo_mass >= 1e11) & (halo_mass <= 3.16e11))}")

## actual driver code

if __name__ == "__main__":
    # Base paths for different simulations
    base_paths = {
        'gauss': '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/HALOS-b0168',
        's8l': '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0_s8l/HALOS-b0168',
        's8h': '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0_s8h/HALOS-b0168',
        'fnl1': '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_fnl1_tpm_seed0/HALOS-b0168',
        'fnl10': '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_fnl10_tpm_seed0/HALOS-b0168'
    }
    
    # Output directory for plots
    plot_dir = './hmf_plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot halo mass functions for all simulations
    plt.figure(figsize=(12, 8))
    
    for sim_name, base_path in base_paths.items():
        print(f"\nAnalyzing simulation: {sim_name}")
        
        # Read halos for step 624 (z=0) without mass cuts
        halo_file = f"{base_path}/m000-624.haloproperties"
        data = read_halofile(halo_file, massbin_min=0.0, massbin_max=1e16)  # Remove mass cuts
        
        # Plot mass function for this simulation
        halo_mass = data["sod_halo_mass"]
        
        # Create mass bins in log space
        mass_bins = np.logspace(np.log10(halo_mass.min()), np.log10(halo_mass.max()), 30)
        
        # Calculate the volume of the simulation box
        boxsize = 2000.0  # Mpc/h
        volume = boxsize**3  # (Mpc/h)^3
        
        # Calculate the halo mass function
        hist, bin_edges = np.histogram(halo_mass, bins=mass_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        dlogM = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        
        # Convert to dn/dlogM
        hmf = hist / (volume * dlogM)
        
        # Plot this simulation's HMF
        plt.loglog(bin_centers, hmf, 'o-', label=f'{sim_name}', alpha=0.7)
        
        # Print statistics
        print(f"\nHalo Mass Function Statistics for {sim_name}:")
        print(f"Total number of halos: {len(halo_mass)}")
        print(f"Mass range: {halo_mass.min():.2e} - {halo_mass.max():.2e} M☉/h")
        print(f"Number of halos > 1e11 M☉/h: {np.sum(halo_mass > 1e11)}")
        print(f"Number of halos 1e11-3.16e11 M☉/h: {np.sum((halo_mass >= 1e11) & (halo_mass <= 3.16e11))}")
    
    # Add vertical lines for mass bins of interest
    plt.axvline(x=1e11, color='r', linestyle='--', label='1e11 M☉/h')
    plt.axvline(x=3.16e11, color='r', linestyle=':', label='3.16e11 M☉/h')
    
    # Customize plot
    plt.xlabel('Halo Mass [M☉/h]')
    plt.ylabel('dn/dlogM [(Mpc/h)⁻³]')
    plt.title('Halo Mass Functions Comparison')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/halo_mass_functions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nHalo mass function plots have been saved to", plot_dir)