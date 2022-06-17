from pyhdf.HDF import *
from pyhdf.V import *
from pyhdf.VS import *
from pyhdf.SD import *
import numpy as np
def HDFread(filename, variable, Class=None):
    """
    Extract the data for non-scientific data in V mode of hdf file
    """
    hdf = HDF(filename, HC.READ)

    # Initialize the SD, V and VS interfaces on the file.
    sd = SD(filename)
    vs = hdf.vstart()
    v  = hdf.vgstart()

    # Found the class id
    if Class == None:
        ref = v.find('Geolocation Fields') # The default value for Geolocation fields
    else:
        ref = v.find(Class)

    # Open all data of the class
    vg = v.attach(ref)
    # All fields in the class
    members = vg.tagrefs()

    nrecs = []
    names = []
    for tag, ref in members:
        # Vdata tag
        try:
            vd = vs.attach(ref)
        except:
            continue
        # nrecs, intmode, fields, size, name = vd.inquire()
        nrecs.append(vd.inquire()[0])  # number of records of the Vdata
        names.append(vd.inquire()[-1]) # name of the Vdata
        vd.detach()

    idx = names.index(variable)
    var = vs.attach(members[idx][1])
    V   = var.read(nrecs[idx])
    var.detach()
    # Terminate V, VS and SD interfaces.
    v.end()
    vs.end()
    sd.end()
    # Close HDF file.
    hdf.close()

    return np.array(V)