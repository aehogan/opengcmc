#!/usr/bin/python3

from opengcmc import GCMCSystem, H2, Ne
from openmm.unit import *

system = GCMCSystem(ensemble=GCMCSystem.uvt, xyz_filename="out.xyz", insert_mol=Ne, freq=100)
system.load_material_xyz("hkust1.xyz")
system.create_openmm_context()
system.build_isotherm()
