#!/usr/bin/python3

from opengcmc import GCMCSystem, H2
from openmm.unit import *

system = GCMCSystem()
system.load_material_xyz("hkust1.xyz")
for _ in range(100):
    system.add_sorbate(H2)
system.dt = 0.002 * picoseconds
system.temperature = 50 * kelvin
# system.ensemble = GCMCSystem.nve
system.create_openmm_context()
# system.freq = 100
# system.step(1000)
system.freq = 10
system.hybrid_mc_step(1000)
