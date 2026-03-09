import numpy as np
import os

# function use to calculate resolution of the filter
def calculate_original_resolution(wavelength):         
    wavelength_array = np.array(wavelength)
    N = len(wavelength_array)
    delta_wl = np.abs((wavelength_array[-1] - wavelength_array[0])) / (N - 1) 
    wl_mid = np.median(wavelength_array) 
    R_orig = wl_mid / delta_wl 
    return R_orig

# function use to change from flux to surface brightness
def flux_to_surface_brightness(I, z):
    I_0 = 1 / 9
    I_result = 1.56e-6 * (I / I_0) * ((1 + z) / 10)**(-4)
    return I_result

# function use to change from flux to magnitude
def flux_to_magnitude(I_nu, z):
    mag_result = -2.5 * np.log10(I_nu) + 8.9
    return mag_result

# function use to calculate width of your filter
def calculate_filter_width(filter_path):
    if not os.path.exists(filter_path):
        print(f"  warning: filter file {filter_path} not found")
        return None
        
    filter_data = np.loadtxt(filter_path, skiprows = 1)
    wavelengths = filter_data[:, 0]
    transmission = filter_data[:, 1]
        
    mask = transmission > 0
    if np.sum(mask) == 0:
        print(f"  warning: no transmission > 0 in filter {filter_path}")
        return None
        
    wl_min = wavelengths[mask].min()
    wl_max = wavelengths[mask].max()
    delta_wl_micron = wl_max - wl_min
    delta_wl_angstrom = delta_wl_micron * 1e4
        
    return delta_wl_angstrom