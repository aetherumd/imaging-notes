import os
import numpy as np
import pandas as pd
from yt import load
from library.filter_tools import create_multiple_filter_files

# function to prepare simulation data
def prepare_simulation_data(input_path, cell_fields = None, epf = None, 
                            filter_path = None, z = 10, 
                            filter_dir = None, wl_initial = None, 
                            wl_final = None, num_bins = 20, 
                            jwst_filter_file = "F200W_filter.txt"):
    """"
    Prepare simulation dataset, star particle positions, and load filter data. This function assumes that you already have input data.

    Parameters:
    ----------
    - input_path (str): path to the simulation dataset file
    - cell_fields (list or None): list of cell fields to load; if None, uses default fields
    - epf (list or None): extra particle fields to load; if None, uses default fields
    - filter_path (str or None): 'F200W_filter.txt' to load JWST filter, None to create filters automatically
    - z (float): redshift to shift filter wavelengths to match galaxy rest frame
    - filter_dir (str): directory for filter bins (default: 'filter_bins' in current directory)
    - wl_initial (float or None): starting wavelength in microns; if None, tracks from jwst_filter_file
    - wl_final (float or None): ending wavelength in microns; if None, tracks from jwst_filter_file
    - num_bins (int): number of filters to create
    - jwst_filter_file (str): JWST filter for resolution reference and auto-limit tracking

    Returns:
    -------
    - ds: yt dataset object
    - ad: all data container from dataset
    - ctr_at_code (np.ndarray): center of star particles in code units (3-element array)
    - pop2_xyz (np.ndarray): star particle positions centered and converted to parsecs
    - wavelength_filter_shifted (np.ndarray): wavelength arrays (Å) for filters, shifted by redshift
    - output_filter (np.ndarray): transmission arrays for filters
    - plt_wdth (int): plot width in parsecs for yt visualization
    """

    # set default filter directory to current working directory if none provided
    if filter_dir is None:
        filter_dir = os.path.join(os.getcwd(), "filter_bins")

    # default fields
    if cell_fields is None:
        cell_fields = [
            "Density", "x-velocity", "y_velocity", "z-velocity", "Pressure",
            "Metallicity", "xHI", "xHII", "xHeII", "xHeIII"
        ]

    if epf is None:
        epf = [
            ("particle_family", "b"),
            ("particle_tag", "b"),
            ("particle_birth_epoch", "d"),
            ("particle_metallicity", "d"),
        ]

    # load dataset
    ds = load(input_path, fields = cell_fields, extra_particle_fields = epf, default_species_fields = "ionized")
    ad = ds.all_data()

    # star particle positions
    x_pos = np.array(ad["star", "particle_position_x"])
    y_pos = np.array(ad["star", "particle_position_y"])
    z_pos = np.array(ad["star", "particle_position_z"])
    x_center, y_center, z_center = np.mean(x_pos), np.mean(y_pos), np.mean(z_pos)
    ctr_at_code = np.array([x_center, y_center, z_center])

    # center positions
    x_pos -= x_center
    y_pos -= y_center
    z_pos -= z_center
    pop2_xyz = np.array(ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")).T

    # define variable to save wavelength and transmission profile of the shifted filter
    wavelength_filter_shifted = []
    output_filter = []

    # handle automatic tracking of wavelength limits if not provided
    if wl_initial is None or wl_final is None:
        if os.path.isfile(jwst_filter_file):
            ref_data = np.loadtxt(jwst_filter_file, skiprows = 1)
            ref_mask = ref_data[:, 1] > 0
            if wl_initial is None:
                wl_initial = ref_data[:, 0][ref_mask].min()
            if wl_final is None:
                wl_final = ref_data[:, 0][ref_mask].max()
            print(f"Auto-tracking limits from {jwst_filter_file}: {wl_initial:.4f}-{wl_final:.4f} microns")
        else:
            # fallback defaults if tracking file is missing
            wl_initial = 0.6 if wl_initial is None else wl_initial
            wl_final = 2.3 if wl_final is None else wl_final

    # create the filter for users if they don't have it
    if filter_path is None:
        print(f"No filter file provided... Creating filters automatically in {filter_dir}")

        bin_edge, filter_files = create_multiple_filter_files(
            filter_dir, wl_initial, wl_final, num_bins, jwst_filter_file
        )

        for i, f in enumerate(filter_files):
            filter_data = np.loadtxt(f, skiprows = 1)
            df_filter = pd.DataFrame(filter_data, columns = ["Wavelength [Microns]", "Output"])
            
            # convert microns to Angstroms
            # shift by redshift to match galaxy rest frame
            shifted_wl = df_filter["Wavelength [Microns]"] * 1e4 / (1 + z)
            
            wavelength_filter_shifted.append(shifted_wl.values)
            output_filter.append(df_filter["Output"].values)
            
            # print the actual range for this filter
            print(f"Loaded filter {i+1}: {os.path.basename(f)}")
            print(f"  Shifted range (z = {z}): {shifted_wl.min():.1f}-{shifted_wl.max():.1f} Å")
        
        # convert lists to arrays
        wavelength_filter_shifted = np.array(wavelength_filter_shifted)
        output_filter = np.array(output_filter)

    # use the filter you have
    elif filter_path == "F200W_filter.txt": 
        if not os.path.isfile(filter_path):
            raise FileNotFoundError(f"Filter file '{filter_path}' does not exist.")

        print(f"Loading filter data from file: {filter_path}")
        filter_data = np.loadtxt(filter_path, skiprows = 1)
        df_filter = pd.DataFrame(filter_data, columns = ["Wavelength [Microns]", "Output"])
        wavelength_filter_shifted = df_filter["Wavelength [Microns]"].values * 1e4 / (1 + z)
        output_filter = df_filter["Output"].values
        
        print(f"Filter wavelength range: {wavelength_filter_shifted.min():.1f}-{wavelength_filter_shifted.max():.1f} Å")

    else:
        raise ValueError(f"Invalid filter_path '{filter_path}'.")

    plt_wdth = 400  # plot width in parsecs for visualization

    return ds, ad, ctr_at_code, pop2_xyz, wavelength_filter_shifted, output_filter, plt_wdth