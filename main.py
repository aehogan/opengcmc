#!/usr/bin/python3

from opengcmc import GCMCSystem
from openmm.unit import *

system = GCMCSystem()
system.load_material_xyz("hkust1.xyz")
system.add_sorbate("H2")
system.temperature = 50 * kelvin
system.create_openmm_context()
system.freq = 100
system.step(1000)
