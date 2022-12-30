#!/usr/bin/python3

from openmm import System, NonbondedForce, CustomNonbondedForce,\
    AmoebaMultipoleForce, LangevinMiddleIntegrator, Context
import sys
import os
import numpy as np

try:
    from opengcmc import Atom, Molecule
except ImportError:
    sys.path.append(os.getcwd())
    from opengcmc import Atom, Molecule

from openmm.unit import *


def load_file(filename):
    with open(filename, "r") as f:
        lines = [line.split() for line in f.readlines()]
        for i, line in enumerate(lines):
            if i == 0:
                mof = Molecule(name="MOF")
                continue
            elif i == 1:
                a = float(line[0])
                b = float(line[1])
                c = float(line[2])
                continue
            name = line[0]
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            if len(line) >= 5:
                charge = float(line[4])
            else:
                charge = 0
            atom = Atom(x, y, z, name, i - 2, charge=charge)
            mof.append(atom)
    positions = np.zeros((len(mof), 3))
    for i, atom in enumerate(mof):
        positions[i] = atom.x
    positions *= nanometers/10.0
    box_vectors = [[a, 0, 0], [0, b, 0], [0, 0, c]] * nanometers/10
    return mof, positions, box_vectors


def add_frozen_molecule_to_openmm_system(system, mol):
    for atom in mol:
        system.addParticle(0)
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            for i, atom in enumerate(mol):
                force.addParticle(atom.charge, atom.lj_sigma, atom.lj_epsilon)
                for j, atom2 in enumerate(mol):
                    if i <= j:
                        continue
                    force.addException(i, j, 0, 0, 0)
        elif isinstance(force, CustomNonbondedForce):
            for i, atom in enumerate(mol):
                force.addParticle((atom.c6, atom.c8, atom.c10, atom.beta, atom.rho))
                for j, atom2 in enumerate(mol):
                    if i <= j:
                        continue
                    force.addExclusion(i, j)
        elif isinstance(force, AmoebaMultipoleForce):
            for i, atom in enumerate(mol):
                force.addMultipole(atom.charge,
                                   (0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                   AmoebaMultipoleForce.NoAxisType,
                                   -1, -1, -1,
                                   0.39,  # Thole damping parameter
                                   atom.alpha ** (1/6),
                                   atom.alpha
                                   )
                other_atoms = [j for j, atom2 in enumerate(mol) if i != j]
                force.setCovalentMap(i, AmoebaMultipoleForce.Covalent12, other_atoms)
                force.setCovalentMap(i, AmoebaMultipoleForce.PolarizationCovalent11, other_atoms)


def create_openmm_system():
    system = System()
    tt_force = CustomNonbondedForce(
        "repulsion - ttdamp6*c6*invR6 - ttdamp8*c8*invR8 - ttdamp10*c10*invR10;"
        "repulsion = forceAtZero*invbeta*exp(-beta*(r-rho));"
        "ttdamp10 = 1.0 - expbr * ttdamp10Sum;"
        "ttdamp8 = 1.0 - expbr * ttdamp8Sum;"
        "ttdamp6 = 1.0 - expbr * ttdamp6Sum;"
        "ttdamp10Sum = ttdamp8Sum + br9/362880 + br10/3628800;"
        "ttdamp8Sum = ttdamp6Sum + br7/5040 + br8/40320;"
        "ttdamp6Sum = 1.0 + br + br2/2 + br3/6 + br4/24 + br5/120 + br6/720;"
        "expbr = exp(-br);"
        "br10 = br5*br5;"
        "br9 = br5*br4;"
        "br8 = br4*br4;"
        "br7 = br4*br3;"
        "br6 = br3*br3;"
        "br5 = br3*br2;"
        "br4 = br2*br2;"
        "br3 = br2*br;"
        "br2 = br*br;"
        "br = beta*r;"
        "invR10 = invR6*invR4;"
        "invR8 = invR4*invR4;"
        "invR6 = invR4*invR2;"
        "invR4 = invR2*invR2;"
        "invR2 = invR*invR;"
        "invR = 1.0/r;"
        "invbeta = 1.0/beta;"
        "c6 = sqrt(c61*c62);"
        "c8 = sqrt(c81*c82);"
        "c10 = sqrt(c101*c102);"
        "beta = 2.0*beta1*beta2/(beta1+beta2);"
        "rho = 0.5*(rho1+rho2);"
    )
    tt_force.addPerParticleParameter("c6")
    tt_force.addPerParticleParameter("c8")
    tt_force.addPerParticleParameter("c10")
    tt_force.addPerParticleParameter("beta")
    tt_force.addPerParticleParameter("rho")
    tt_force.addGlobalParameter("forceAtZero", 49.6144931952)  # kJ/(mol*A)
    tt_force.setCutoffDistance(0.9)
    tt_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    tt_force.setUseLongRangeCorrection(False)
    system.addForce(tt_force)
    mp_force = AmoebaMultipoleForce()
    mp_force.setNonbondedMethod(AmoebaMultipoleForce.PME)
    mp_force.setPolarizationType(AmoebaMultipoleForce.Extrapolated)
    mp_force.setCutoffDistance(0.9)
    system.addForce(mp_force)
    return system


mof, positions, box_vectors = load_file("hkust1.xyz")
openmm_system = create_openmm_system()
add_frozen_molecule_to_openmm_system(openmm_system, mof)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
context = Context(openmm_system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(298)
context.setPeriodicBoxVectors(*box_vectors)
integrator.step(1000)
