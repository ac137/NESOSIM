import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from config import forcing_save_path


# use np.load to read files

#'/users/jk/18/acabaj/NESOSIM/Forcings/Precip/ERAI'


#'/users/jk/18/acabaj/NESOSIM/Forcings/Winds/ERAI/2010'

# filename format: ERAIwinds50km-2010_d270v11
# or ERAIwinds50km-2010_d000v11

# for precip: ERA5sf50km-2010_d000v11 (for example)
year = 2010
ei_precip_path = forcing_save_path + '/Precip/ERAI/{}/'.format(year)
e5_precip_path = forcing_save_path + '/Precip/ERA5/{}/'.format(year)

sf_ei = np.load(ei_precip_path + 'ERAIsf50km-2010_d000v11')
sf_e5 = np.load(e5_precip_path + 'ERA5sf50km-2010_d000v11')

# this isn't masked over land, though
print(sf_ei)
print(sf_ei.shape)