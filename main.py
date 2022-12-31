#!/usr/bin/python3

import sys
import os
from opengcmc import Atom, Molecule, GCMCSystem

system = GCMCSystem()
system.load_material_xyz("hkust1.xyz")
system.add_sorbate("H2")
system.create_openmm_context()
system.freq = 1000
system.step(10000)
