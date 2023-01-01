#!/usr/bin/python3

from opengcmc import GCMCSystem

system = GCMCSystem()
system.load_material_xyz("hkust1.xyz")
system.add_sorbate("H2")
system.create_openmm_context()
system.freq = 100
system.step(1000)
