#!/usr/bin/python3

from opengcmc import GCMCSystem, H2
from openmm.unit import *

system = GCMCSystem(ensemble=GCMCSystem.uvt, xyz_filename="out.xyz", insert_mol=H2, pressure=1000*atmospheres)
system.load_material_xyz("hkust1.xyz")
for _ in range(10):
    system.add_sorbate()
# system.fill_with_sorbate()
system.create_openmm_context()
system.freq = 100
system.step(1000)
