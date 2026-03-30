import numpy as np
import yt
from yt.frontends.ramses.field_handlers import RTFieldFileHandler
from merlinconstants import default_lines, default_epf
from merlin_spectra.emission import EmissionLineInterpolator

class Initializer:
    def __init__(self, ds = None, ds_file = None, lines = default_lines, epf = default_epf):
        """
        Initializes the initalizer by providing a dataset file to be loaded with yt. 
        
        Parameters:
            ds_file (String): The path to the dataset file to be loaded by yt.
            lines (List[String]): A list of lines to be created from the EmissionLineInterpolator
            epf (List[Tuple[str, str]]): A list of tuple specifying extra particle fields to be loaded by yt
        """
        if ds is not None:
            self.ds = ds
        else:
            self.ds = self._load_data(ds_file, epf)
        self.lines = lines

    def _load_data(self, ds_file, epf):
        """
        Loads the dataset file using yt
        
        Parameters:
            ds_file (String): The path to the dataset file to be loaded by yt.
            epf (List[Tuple[str, str]]): A list of tuples specifying the extra particle fields to load.
        """
        return yt.load(ds_file, extra_particle_fields=epf)

    def _load_fields(self):
        self.ds.add_field(
            ("gas","number_density"),
            function=_my_H_nuclei_density,
            sampling_type="cell",
            units="1/cm**3",
            force_override=True
        )

        self.ds.add_field(
            ("ramses","Pressure"),
            function=_pressure,
            sampling_type="cell",
            units="1",
            #force_override=True
        )

        self.ds.add_field(
            ("ramses","xHI"),
            function=_xHI,
            sampling_type="cell",
            units="1",
            #force_override=True
        )

        self.ds.add_field(
            ("ramses","xHII"),
            function=_xHII,
            sampling_type="cell",
            units="1",
            #force_override=True
        )

        self.ds.add_field(
            ("ramses","xHeII"),
            function=_xHeII,
            sampling_type="cell",
            units="1",
            #force_override=True
        )

        self.ds.add_field(
            ("ramses","xHeIII"),
            function=_xHeIII,
            sampling_type="cell",
            units="1",
            #force_override=True
        )

        self.ds.add_field(
            ("gas","my_temperature"),
            function=_my_temperature,
            sampling_type="cell",
            # TODO units
            #units="K",
            #units="K*cm**3/erg",
            units='K*cm*dyn/erg',
            force_override=True
        )

        # Ionization parameter
        self.ds.add_field(
            ('gas', 'ion_param'),
            function=_ion_param,
            sampling_type="cell",
            units="cm**3",
            force_override=True
        )

        self.ds.add_field(
            ("gas","my_H_nuclei_density"),
            function=_my_H_nuclei_density,
            sampling_type="cell",
            units="1/cm**3",
            force_override=True
        )

    def _load_luminosity_flux(self, dens_normalized=False):
        if dens_normalized: 
            flux_units = '1/cm**6'
            lum_units = '1/cm**3'
        else:
            flux_units = '1'
            lum_units = 'cm**3'

        emission_interpolator = EmissionLineInterpolator(lines=self.lines)

        # Add flux and luminosity fields for all lines in the list
        for i, line in enumerate(self.lines):
            self.ds.add_field(
                ('gas', 'flux_' + line),
                function=emission_interpolator.get_line_emission(
                    i, dens_normalized=dens_normalized
                ),
                sampling_type='cell',
                units=flux_units,
                force_override=True,
            )

            self.ds.add_field(
                ('gas', 'luminosity_' + line),
                function=emission_interpolator.get_luminosity(self.lines[i]),
                sampling_type='cell',
                units=lum_units,
                force_override=True,
            )

        self.ds.add_field(
            ("gas","OII_ratio"),
            function=_OII_ratio,
            sampling_type="cell",
            units="1",
            force_override=True
        )

# All of these functions are defined to create the derived fields needed to use the line emission interpolator, which can
# get the luminosity of each line.

# Ionization Parameter Field
# Based on photon densities in bins 2-4
# Don't include bin 1 -> Lyman Werner non-ionizing
def _ion_param(field, data):
    p = RTFieldFileHandler.get_rt_parameters(data.ds).copy()
    p.update(data.ds.parameters)

    cgs_c = 2.99792458e10     #light velocity

    # Convert to physical photon number density in cm^-3
    pd_2 = data['ramses-rt','Photon_density_2']*p["unit_pf"]/cgs_c
    pd_3 = data['ramses-rt','Photon_density_3']*p["unit_pf"]/cgs_c
    pd_4 = data['ramses-rt','Photon_density_4']*p["unit_pf"]/cgs_c

    photon = pd_2 + pd_3 + pd_4

    return photon/data['gas', 'number_density']


def _my_temperature(field, data):
    #y(i): abundance per hydrogen atom
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90
    kB_RAMSES=yt.YTArray(1.3806200e-16,"erg/K") #defined by RAMSES in cooling_module.f90

    dn=data["ramses","Density"].in_cgs()
    pr=data["ramses","Pressure"].in_cgs()
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yHe = YHE_RAMSES*0.25/XH_RAMSES
    yHeII=data["ramses","xHeII"]*yHe
    yHeIII=data["ramses","xHeIII"]*yHe
    yH2=1.-yHI-yHII
    yel=yHII+yHeII+2*yHeIII
    mu=(yHI+yHII+2.*yH2 + 4.*yHe) / (yHI+yHII+yH2 + yHe + yel)
    return pr/dn * mu * mH_RAMSES / kB_RAMSES


#number density of hydrogen atoms
def _my_H_nuclei_density(field, data):
    dn=data["ramses","Density"].in_cgs()
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90

    return dn*XH_RAMSES/mH_RAMSES


def _pressure(field, data):
    if 'hydro_thermal_pressure' in dir(data.ds.fields.ramses): # and 
        #'Pressure' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_thermal_pressure']


def _xHI(field, data):
    if 'hydro_xHI' in dir(data.ds.fields.ramses): # and \
        #'xHI' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHI']


def _xHII(field, data):
    if 'hydro_xHII' in dir(data.ds.fields.ramses): # and \
        #'xHII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHII']


def _xHeII(field, data):
    if 'hydro_xHeII' in dir(data.ds.fields.ramses): # and \
        #'xHeII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHeII']


def _xHeIII(field, data):
    if 'hydro_xHeIII' in dir(data.ds.fields.ramses): # and \
        #'xHeIII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHeIII']
    
def _OII_ratio(field, data):
    # TODO lum or flux?
    #return data['gas', 'flux_O2_3728.80A']/data['gas', 'flux_O2_3726.10A']
    flux1 = data['gas', 'flux_O2_3728.80A']
    flux2 = data['gas', 'flux_O2_3726.10A']

    flux2 = np.where(flux2 < 1e-30, 1e-30, flux2)

    ratio = flux1 / flux2

    return ratio