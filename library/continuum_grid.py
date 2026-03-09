import os
import numpy as np
import pandas as pd
from pyneb import Continuum
from scipy.interpolate import interp1d, LinearNDInterpolator

# function to calculate continuum grid
def compute_continuum_grid(min_temp, max_temp, num_temp_grid, min_dense, max_dense, num_dense_grid, min_wl, max_wl, num_wl_grid, filter_wl = None, filter_output = None, save_dir = None):
    """
    Compute continuum flux (bound-free, two-photon, free-free) over a grid of temperature and number density, which will be used as a reference to calculate flux value of each pixel in the real object

    Parameters:
    -----------
    - min_temp (float): minimum temperature in your grid (in K)
    - max_temp (float): maximum temperature in your grid (in K)
    - num_temp_grid (int): number of data points in temperature grid
    - min_dense (float): minimum number density in your grid (in cm^-3)
    - max_dense (float): maximum number density in your grid (in cm^-3)
    - num_dense_grid (int): number of data points in density grid
    - min_wl (float): minimum wavelength of your pixel (in A). Keep in mind that this value for continuum cannot be smaller than 912 A
    - max_wl (float): maximum wavelength of your pixel (in A). Keep in mind that this value for continuum cannot be larger than 1e5 A
    - num_wl_grid (int): number of data points in wavelength grid
    - filter_wl (array): filter's wavelength range, which is what you prepare using prepare_simulation_data function
        - 1D array for single filter
        - 2D array (n_filters, n_wavelength_points) for multiple filters
    - filter_output (array): filter's transmission profile within your filter range, which is what you prepare using prepare_simulation_data function
        - 1D array for single filter
        - 2D array (n_filters, n_wavelength_points) for multiple filters
    - save_dir (str or None): directory to save dataframe files. If None, creates 'df_filter' folder in current directory.

    Outputs:
    --------
    - df_results (DataFrame or dict): 
        - single filter: DataFrame with T, n, and flux averages
        - multiple filters: dictionary of DataFrames, keyed by filter index
    - interp_funcs (dict or dict of dicts):
        - single filter: dictionary of LinearNDInterpolator for each component
        - multiple filters: dictionary of dictionaries, where each key is filter index and value is dict of interpolators
    """

    # build the grid for temperature, density, and wavelength
    temperature_grid = np.logspace(np.log10(min_temp), np.log10(max_temp), num_temp_grid)
    density_grid = np.logspace(np.log10(min_dense), np.log10(max_dense), num_dense_grid)
    wl = np.logspace(np.log10(min_wl), np.log10(max_wl), num_wl_grid)

    # create a continuum using Pyneb package
    C = Continuum()
    
    # check if we have multiple filters
    filter_wl_array = np.asarray(filter_wl) if filter_wl is not None else None
    filter_output_array = np.asarray(filter_output) if filter_output is not None else None
    
    # if we have 2D arrays (multiple filters), process them one by one
    if filter_wl_array.ndim == 2 and filter_output_array.ndim == 2:
        # multiple filters case
        n_filters = filter_wl_array.shape[0]
        print(f"Processing {n_filters} filters...")
        
        # create directory for saving dataframes
        if save_dir is None:
            save_dir = "df_filter"
        os.makedirs(save_dir, exist_ok = True)
        print(f"Saving dataframes to: {save_dir}")
        
        all_df_results = {}
        all_interp_funcs = {}
        
        for filter_idx in range(n_filters):
            filter_number = filter_idx + 1  # 1-based indexing
            print(f"\nFilter {filter_number}/{n_filters}:")
            
            # get current filter data
            current_filter_wl = filter_wl_array[filter_idx]
            current_filter_output = filter_output_array[filter_idx]
            
            # keep only points within continuum wavelength range AND with transmission > 0
            mask = (current_filter_wl >= min_wl) & (current_filter_wl <= max_wl) & (current_filter_output > 0)
            calc_filter_wl = current_filter_wl[mask]
            calc_filter_output = current_filter_output[mask]
            
            if len(calc_filter_wl) == 0:
                print(f"  No points in range {min_wl}-{max_wl} Å with transmission > 0, skipping...")
                # create empty dataframe for this filter
                df_results = pd.DataFrame()
                filter_name = f"filter_{filter_number:02d}"
                all_df_results[filter_name] = df_results
                
                # save empty dataframe to file
                filename = f"df_{filter_number:02d}_empty.txt"
                filepath = os.path.join(save_dir, filename)
                df_results.to_csv(filepath, sep = '\t', index = False)
                print(f"  Saved: {filename} (empty)")
                continue
            
            # check if filter range overlaps with continuum wavelength grid
            filter_min = calc_filter_wl.min()
            filter_max = calc_filter_wl.max()
            
            if filter_max < wl.min() or filter_min > wl.max():
                print(f"  Filter range {filter_min:.1f}-{filter_max:.1f} Å outside continuum grid {wl.min():.1f}-{wl.max():.1f} Å")
                print(f"  Will produce NaN values")
                # still process but will get NaN
            
            print(f"  Filter range: {filter_min:.1f}-{filter_max:.1f} Å")
            print(f"  Points with transmission > 0: {len(calc_filter_wl)}")
            
            # run single filter calculation
            results = []
            
            for tem in temperature_grid:
                for den in density_grid:
                    contH = C.get_continuum(tem = tem, den = den, wl = wl, HI_label = None, cont_HI = True, cont_HeI = True, cont_HeII = True, cont_2p = False, cont_ff = False)
                    cont2p = C.get_continuum(tem = tem, den = den, wl = wl, HI_label = None, cont_HI = False, cont_HeI = False, cont_HeII = False, cont_2p = True, cont_ff = False)
                    contff = C.get_continuum(tem = tem, den = den, wl = wl, HI_label = None, cont_HI = False, cont_HeI = False, cont_HeII = False, cont_2p = False, cont_ff = True)
                    
                    # check if filter range overlaps with continuum grid
                    if filter_max < wl.min() or filter_min > wl.max():
                        # no overlap - produce NaN
                        flux_avg_H = np.nan
                        flux_avg_2p = np.nan
                        flux_avg_ff = np.nan
                    else:
                        # find overlapping region
                        mask_wl = (wl >= max(filter_min, wl.min())) & (wl <= min(filter_max, wl.max()))
                        
                        if mask_wl.sum() == 0:
                            flux_avg_H = np.nan
                            flux_avg_2p = np.nan
                            flux_avg_ff = np.nan
                        else:
                            fluxH_interp = interp1d(wl[mask_wl], contH[mask_wl]*wl[mask_wl], fill_value = "extrapolate")
                            flux2p_interp = interp1d(wl[mask_wl], cont2p[mask_wl]*wl[mask_wl], fill_value = "extrapolate")
                            fluxff_interp = interp1d(wl[mask_wl], contff[mask_wl]*wl[mask_wl], fill_value = "extrapolate")
                            
                            resultH = fluxH_interp(calc_filter_wl) * calc_filter_output
                            result2p = flux2p_interp(calc_filter_wl) * calc_filter_output
                            resultff = fluxff_interp(calc_filter_wl) * calc_filter_output
                            integral_filter = np.trapz(calc_filter_output * calc_filter_wl, calc_filter_wl)
                            
                            if integral_filter > 0:
                                flux_avg_H = np.trapz(resultH, calc_filter_wl) / integral_filter
                                flux_avg_2p = np.trapz(result2p, calc_filter_wl) / integral_filter
                                flux_avg_ff = np.trapz(resultff, calc_filter_wl) / integral_filter
                            else:
                                flux_avg_H = np.nan
                                flux_avg_2p = np.nan
                                flux_avg_ff = np.nan
                    
                    results.append({
                        "Temperature": tem,
                        "Density": den,
                        "Average Specific Flux ContH": flux_avg_H,
                        "Average Specific Flux Cont2p": flux_avg_2p,
                        "Average Specific Flux Contff": flux_avg_ff
                    })
            
            # create DataFrame for this filter
            df_results = pd.DataFrame(results)
            
            # check for NaN values
            nan_count = df_results[["Average Specific Flux ContH", "Average Specific Flux Cont2p", "Average Specific Flux Contff"]].isna().sum().sum()
            if nan_count > 0:
                print(f"  Contains {nan_count} NaN values")
            
            # remove rows with NaN values for interpolation
            df_results_clean = df_results.dropna()
            
            # store with filter name/key (1-based: 01, 02, ...)
            filter_name = f"filter_{filter_number:02d}"
            all_df_results[filter_name] = df_results
            
            # create interpolation functions if we have valid data
            if len(df_results_clean) > 0:
                points = np.column_stack((df_results_clean["Temperature"], df_results_clean["Density"]))
                interp_funcs = {
                    "contH": LinearNDInterpolator(points, df_results_clean["Average Specific Flux ContH"]),
                    "cont2p": LinearNDInterpolator(points, df_results_clean["Average Specific Flux Cont2p"]),
                    "contff": LinearNDInterpolator(points, df_results_clean["Average Specific Flux Contff"]),
                }
                all_interp_funcs[filter_name] = interp_funcs
            
            # save dataframe to file
            # create filename with wavelength range
            filename = f"df_{filter_number:02d}_wl{filter_min:.0f}-{filter_max:.0f}A.txt"
            filepath = os.path.join(save_dir, filename)
            
            # save dataframe to txt file
            df_results.to_csv(filepath, sep = '\t', index = False)
            print(f"  Saved: {filename} ({len(df_results)} rows)")
        
        print(f"\nCreated {len(all_df_results)} dataframes and {len(all_interp_funcs)} interpolation functions")
        print(f"All dataframes saved to: {save_dir}")
        
        # create variable for each dataframe
        print("\nCreating variables df01, df02, etc. for available filters:")
        for i in range(1, n_filters + 1):
            filter_key = f"filter_{i:02d}"
            var_name = f"df{i:02d}"
            
            if filter_key in all_df_results:
                df = all_df_results[filter_key]
                if not df.empty:
                    globals()[var_name] = df
                    print(f"  Created {var_name} = shape {df.shape}")
        
        return all_df_results, all_interp_funcs
        
    else:
        # single filter case (1D array)
        print("Processing single filter...")
        
        # create directory for saving dataframes
        if save_dir is None:
            save_dir = "df_filter"
        os.makedirs(save_dir, exist_ok = True)
        print(f"Saving dataframe to: {save_dir}")
        
        # ensure 1D arrays
        filter_wl_1d = filter_wl_array.flatten()
        filter_output_1d = filter_output_array.flatten()
        
        # keep only points within continuum wavelength range AND with transmission > 0
        mask = (filter_wl_1d >= min_wl) & (filter_wl_1d <= max_wl) & (filter_output_1d > 0)
        calc_filter_wl = filter_wl_1d[mask]
        calc_filter_output = filter_output_1d[mask]
        
        if len(calc_filter_wl) == 0:
            raise ValueError(f"Filter has no points in {min_wl}–{max_wl} Å range with transmission > 0")
        
        filter_min = calc_filter_wl.min()
        filter_max = calc_filter_wl.max()
        print(f"Filter range: {filter_min:.1f}-{filter_max:.1f} Å")
        print(f"Points with transmission > 0: {len(calc_filter_wl)}")
        
        # check if filter range overlaps with continuum grid
        if filter_max < wl.min() or filter_min > wl.max():
            print(f"WARNING: Filter range outside continuum grid {wl.min():.1f}-{wl.max():.1f} Å")
        
        results = []
        
        for tem in temperature_grid:
            for den in density_grid:
                contH = C.get_continuum(tem = tem, den = den, wl = wl, HI_label = None, cont_HI = True, cont_HeI = True, cont_HeII = True, cont_2p = False, cont_ff = False)
                cont2p = C.get_continuum(tem = tem, den = den, wl = wl, HI_label = None, cont_HI = False, cont_HeI = False, cont_HeII = False, cont_2p = True, cont_ff = False)
                contff = C.get_continuum(tem = tem, den = den, wl = wl, HI_label = None, cont_HI = False, cont_HeI = False, cont_HeII = False, cont_2p = False, cont_ff = True)
                
                # apply filter
                mask_wl = (wl >= max(filter_min, wl.min())) & (wl <= min(filter_max, wl.max()))
                fluxH_interp = interp1d(wl[mask_wl], contH[mask_wl]*wl[mask_wl], fill_value = "extrapolate")
                flux2p_interp = interp1d(wl[mask_wl], cont2p[mask_wl]*wl[mask_wl], fill_value = "extrapolate")
                fluxff_interp = interp1d(wl[mask_wl], contff[mask_wl]*wl[mask_wl], fill_value = "extrapolate")
                
                resultH = fluxH_interp(calc_filter_wl) * calc_filter_output
                result2p = flux2p_interp(calc_filter_wl) * calc_filter_output
                resultff = fluxff_interp(calc_filter_wl) * calc_filter_output
                integral_filter = np.trapz(calc_filter_output * calc_filter_wl, calc_filter_wl)
                
                flux_avg_H = np.trapz(resultH, calc_filter_wl) / integral_filter
                flux_avg_2p = np.trapz(result2p, calc_filter_wl) / integral_filter
                flux_avg_ff = np.trapz(resultff, calc_filter_wl) / integral_filter
                
                results.append({
                    "Temperature": tem,
                    "Density": den,
                    "Average Specific Flux ContH": flux_avg_H,
                    "Average Specific Flux Cont2p": flux_avg_2p,
                    "Average Specific Flux Contff": flux_avg_ff
                })
        
        df_results = pd.DataFrame(results)
        
        # save dataframe to file
        filename = f"df_wl{filter_min:.0f}-{filter_max:.0f}A.txt"
        filepath = os.path.join(save_dir, filename)
        df_results.to_csv(filepath, sep = '\t', index = False)
        print(f"Saved: {filename} ({len(df_results)} rows)")
        
        # create df01 variable for single filter
        print("Creating variable df01 for single filter:")
        globals()['df01'] = df_results
        print(f"  Created df01 = shape {df_results.shape}")
        
        points = np.column_stack((df_results["Temperature"], df_results["Density"]))
        interp_funcs = {
            "contH": LinearNDInterpolator(points, df_results["Average Specific Flux ContH"]),
            "cont2p": LinearNDInterpolator(points, df_results["Average Specific Flux Cont2p"]),
            "contff": LinearNDInterpolator(points, df_results["Average Specific Flux Contff"]),
        }
        return df_results, interp_funcs