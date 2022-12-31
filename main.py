#!/usr/bin/python3

import sys
import os

try:
    from opengcmc import Atom, Molecule, GCMCSystem
except ImportError:
    sys.path.append(os.getcwd())
    from opengcmc import Atom, Molecule, GCMCSystem

system = GCMCSystem()
system.load_material_xyz("hkust1.xyz")
system.create_openmm_context()
system.step(100)
