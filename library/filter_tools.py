import os
import numpy as np
from library.calculate_quantities import calculate_original_resolution

# function to create the filter without redshifting
def create_multiple_filter_files(output_dir, wl_initial, wl_final, num_filters = 20, jwst_filter_path = "F200W_filter.txt"):
    """"
    Create multiple filter files. Each filter covers the full wavelength range,
    but transmission = 1 only in its specific bin, 0 elsewhere.

    Parameters:
    ----------
    output_dir (str): location to save these filter files.
    wl_initial (float): initial wavelength of the full wavelength range (in microns).
    wl_final (float): final wavelength of the full wavelength range (in microns).
    num_filters (int): number of filters you want to create.
    jwst_filter_path (str): location of your JWST filter. This will be used for calculating resolution, which will be matched with filter's resolution created using this function. 
    This means that this function requires you to download JWST filter from this link: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters#gsc.tab=0

    Outputs:
    -------
    bin_edge (np.ndarray): wavelength boundaries for each filter bin  
    jwst_filter_paths (str): file paths for each created filter file
    """

    # if you choose to create store this filter, this part will create output directory
    os.makedirs(output_dir, exist_ok = True)

    # store values of JWST filter data
    jwst_data = np.loadtxt(jwst_filter_path, skiprows = 1)
    jwst_wavelength = jwst_data[:, 0]  # wavelength in microns

    # calculate resolution of the jwst filter
    jwst_resolution = calculate_original_resolution(jwst_wavelength) # this resolution will be used as a resolution of each filter

    # wavelength grid of multiple filters to calculate spacing between each data point
    wavelength_filter_grid = np.logspace(np.log10(wl_initial), np.log10(wl_final), num_filters + 1) # this variable will tell you where the bin edge is

    # calculate number of data points in each filter to match with jwst resolution using the first filter
    central_wavelength = (wavelength_filter_grid[0] + wavelength_filter_grid[1])/2
    delta_wavelength = wavelength_filter_grid[1] - wavelength_filter_grid[0]
    num_points = 1 + (delta_wavelength * jwst_resolution)/central_wavelength
    num_points = int(np.round(num_points))
    
    # calculate spacing between data points
    wavelength_spacing = (wavelength_filter_grid[1] - wavelength_filter_grid[0])/(num_points - 1)

    # create wavelength array using the same spacing from calculation above
    wl_array = np.arange(wl_initial, wl_final, wavelength_spacing) # x-axis of all filters

    # empty array of filters
    filter_paths = []

    # for loop to create filters
    for i in range(num_filters):
        # empty array of transmission
        transmission = np.zeros_like(wl_array)

        # consider range of each filter
        mask = (wl_array >= wavelength_filter_grid[i]) & (wl_array < wavelength_filter_grid[i + 1])

        # transmission in the consider range = 1
        transmission[mask] = 1

        # save filter files
        file_name = f"filter_{i+1}.txt"
        file_path = os.path.join(output_dir, file_name)

        # save as two columns: wavelength and transmission
        data_to_save = np.column_stack((wl_array, transmission))
        np.savetxt(file_path, data_to_save, header = "Wavelength[Microns] Transmission", fmt = "%.6f")
        filter_paths.append(file_path)

    print(f"Successfully created {num_filters} filters in {output_dir}")

    return wavelength_filter_grid, filter_paths