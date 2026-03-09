import numpy as np
import yt
import re
from unyt import unyt_array

# Constants used in the function
headers = ["#", "ID", "CurrentAges[MYr]", "X[pc]", "Y[pc]", "Z[pc]", "mass[Msun]", "t_sim[MYr], z, ctr(code), ctr(pc)"]
cgs_yr = 3.1556926e7  # 1yr (in s)
cgs_pc = 3.08567758e18  # pc (in cm)

def convert_to_text(filepath, 
    epf = [
        ("particle_family", "b"),
        ("particle_tag", "b"),
        ("particle_birth_epoch", "d"),
        ("particle_metallicity", "d"),
    ]
):
    ds = yt.load(filepath, extra_particle_fields=epf)

    time = ds.current_time.in_units("Myr").value
    z = ds.current_redshift

    # center of mass for stars in code units
    ad = ds.all_data()
    x1 = ad["star", "particle_position_x"]
    y1 = ad["star", "particle_position_y"]
    z1 = ad["star", "particle_position_z"]

    center_code = np.array([x1.mean(), y1.mean(), z1.mean()])
    center_pc = unyt_array(center_code * ds.length_unit.in_units("pc").value, "pc")

    h_0 = ds.hubble_constant * 100 # hubble parameter (km/s/Mpc)
    h_0_invsec = h_0 * 1e5 / (1e6 * cgs_pc)  # hubble constant h [km/s Mpc-1]->[1/sec]
    h_0inv_yr = 1 / h_0_invsec / cgs_yr  # 1/h_0 [yr]
    stellar_ages = np.array(ad["star", "particle_birth_epoch"]) * h_0inv_yr / 1e6 + 13.787 * 1e3

    x2 = x1.in_units("pc") - center_pc[0]
    y2 = y1.in_units("pc") - center_pc[1]
    z2 = z1.in_units("pc") - center_pc[2]

    # The 8 values in the last column are time, z, center_code(x,y,z), center_pc(x,y,z)
    last_col = [time, z, center_code[0], center_code[1], center_code[2], center_pc[0].value, center_pc[1].value, center_pc[2].value]

    pattern = r".*info_(\d{5})\.txt"
    file_name = "output_" + re.search(pattern, filepath).group(1) + ".txt"

    with open(file_name, "w") as f:
        # Formats the header row
        f.write("\t\t".join(headers) + "\n")
        ###
        # ID is right now 0, need code to randomly assign unique IDs to stars. The star masses are also
        # defaulted to 10 MSuns, might want to investigate randomly assigning masses through Dr. Ricotti's code. 
        ###
        for i in range(len(stellar_ages)):
            # In the last column, only the first 8 rows have information, the rest are 0s.
            if i >= 8:
                last_col_info = 0
            else:
                last_col_info = last_col[i]

            r_i = [0, stellar_ages[i], x2[i].value, y2[i].value, z2[i].value, 10, last_col_info]
            r_sci = ["{:.18e}".format(n) for n in r_i]
            r_str = " ".join(r_sci)
            f.write(r_str + "\n")

    return file_name

convert_to_text('data/output_00273/info_00273.txt')