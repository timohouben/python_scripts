import numpy as np
from ogs5py import MSH

mesh = MSH()
mesh.generate()
cent = mesh.centroids_flat
x = cent[:, 0]
y = cent[:, 1]
mask_1 = np.logical_and(y > 2, y < 4)
print(mask_1)
print(mask_2)
mask_2 = np.logical_and(y > 4, y < 7)
mesh.set_material_id(1, element_mask=mask_1)
mesh.set_material_id(2, element_mask=mask_2)
mesh.show(show_material_id=True)
import sys

print(sys.executable)
