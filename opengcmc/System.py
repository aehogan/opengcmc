import random

import numpy as np
import time
from .Atom import Atom, Molecule
from .Utils import PBC, Modeler
from .ForceField import FF, Sorbate, H2
from openmm import NonbondedForce, CustomNonbondedForce, AmoebaMultipoleForce, \
    System, NoseHooverIntegrator, Context, XmlSerializer, VerletIntegrator
from openmm.unit import *


class Averages:
    def __init__(self):
        self.n = []
        self.u = []


class GCMCSystem:
    # Thermodynamic ensemble enum
    uvt, npt, nvt, nve = range(4)
    ensemble_to_name = {0: "muVT", 1: "NPT", 2: "NVT", 3: "NVE"}

    def __init__(self, ensemble=None, pressure=1.0*atmospheres, temperature=298.0*kelvins, insert_mol=None,
                 dt=0.001*picoseconds, freq=1000, xyz_filename=None):
        if ensemble is None:
            self.ensemble = GCMCSystem.nvt
        else:
            self.ensemble = ensemble
        if ensemble == GCMCSystem.uvt:
            if insert_mol is None:
                self.insert_mol = H2
            else:
                self.insert_mol = insert_mol
        self.pressure = pressure
        self.temperature = temperature
        # Delta time(step)
        self.dt = dt
        # Frequency of output
        self.freq = freq
        # Output f-string
        if self.ensemble == GCMCSystem.uvt:
            self.format_string = "Step {step:6d} PE {pot_energy:10.3f} kJ/mol N {n:4d}"
        else:
            self.format_string = "Step {step:6d} Time {time:6.2f} ps " \
                             "TE {tot_energy:10.3f} kJ/mol " \
                             "KE {kin_energy:6.3f} kJ/mol " \
                             "PE {pot_energy:10.3f} kJ/mol " \
                             "Elapsed time {elapsed_s:5.1f}s - {ns_per_day:5.2f} ns/day"
        # Write .xyz flag/filename
        if xyz_filename is None:
            self.write_xyz = False
            self.xyz_filename = ""
        else:
            self.write_xyz = True
            self.xyz_filename = xyz_filename
        # list of Molecule objects
        self.mols = []
        # Current number of atoms
        self._n = 0
        # Current step number
        self._step = 0

        # internal objects
        self._pbc = None
        self._out_file = None
        self._ff = FF.phahst
        self._start_time = time.perf_counter()
        self._averages = Averages()

        # OpenMM objects
        self._omm_system = None
        self._omm_integrator = None
        self._omm_context = None
        self._omm_state = None
        self._constraints = []

    def create_box(self, a, b, c, alpha, beta, gamma):
        self._pbc = PBC(a, b, c, alpha, beta, gamma)

    def load_material_xyz(self, filename):
        with open(filename, "r") as f:
            lines = [line.split() for line in f.readlines()]
            mof = Molecule(name="MOF", frozen=True)
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                elif i == 1:
                    try:
                        a = float(line[0])
                        b = float(line[1])
                        c = float(line[2])
                        alpha = float(line[3])
                        beta = float(line[4])
                        gamma = float(line[5])
                        self._pbc = PBC(a, b, c, alpha, beta, gamma)
                    except ValueError:
                        if self._pbc is None:
                            self._pbc = PBC(100, 100, 100, 90, 90, 90)
                    continue
                name = line[0]
                x = float(line[1])
                y = float(line[2])
                z = float(line[3])
                if len(line) >= 5:
                    charge = float(line[4])
                else:
                    charge = 0
                atom = Atom(x, y, z, name, self._n, charge=charge)
                self._n += 1
                mof.append(atom)
            FF.apply_ff(mof.atoms, self._ff)
            self.mols.append(mof)

    def add_sorbate(self):
        if not issubclass(self.insert_mol, Sorbate):
            raise Exception("Sorbate not member of Sorbate base class from ForceField.py")
        sorbate = self.insert_mol(self._n)
        while Modeler.overlap_mol_test(sorbate.molecule, self.mols, self._pbc):
            Modeler.move_mol_randomly(sorbate.molecule, self._pbc)
        for constraint in sorbate.constraints:
            self._constraints.append(constraint)
        self._n += len(sorbate.molecule)
        self.mols.append(sorbate.molecule)

    def fill_with_sorbate(self):
        if not issubclass(self.insert_mol, Sorbate):
            raise Exception("Sorbate not member of Sorbate base class from ForceField.py")
        for _ in range(10):
            keep_looping = True
            while keep_looping:
                sorbate = self.insert_mol(self._n)
                trys = 0
                while Modeler.overlap_mol_test(sorbate.molecule, self.mols, self._pbc):
                    Modeler.move_mol_randomly(sorbate.molecule, self._pbc)
                    trys += 1
                    if trys > 1000:
                        keep_looping = False
                        break
                if keep_looping:
                    for constraint in sorbate.constraints:
                        self._constraints.append(constraint)
                    self._n += len(sorbate.molecule)
                    self.mols.append(sorbate.molecule)

    def add_molecules_to_openmm_system(self):
        for mol in self.mols:
            for atom in mol:
                if mol.frozen:
                    self._omm_system.addParticle(0)
                else:
                    self._omm_system.addParticle(atom.mass)
                if atom.virtual:
                    self._omm_system.setVirtualSite(atom.id, atom.virtual_type)
            for force in self._omm_system.getForces():
                if isinstance(force, NonbondedForce):
                    for atom in mol:
                        i = atom.id
                        force.addParticle(atom.charge, atom.lj_sigma, atom.lj_epsilon)
                        for atom2 in mol:
                            j = atom2.id
                            if i <= j:
                                continue
                            force.addException(i, j, 0, 0, 0)
                elif isinstance(force, CustomNonbondedForce):
                    for atom in mol:
                        force.addParticle((atom.c6, atom.c8, atom.c10, atom.beta, atom.rho))
                        i = atom.id
                        for atom2 in mol:
                            j = atom2.id
                            if i <= j:
                                continue
                            force.addExclusion(i, j)
                elif isinstance(force, AmoebaMultipoleForce):
                    for atom in mol:
                        i = atom.id
                        force.addMultipole(atom.charge,
                                           (0.0, 0.0, 0.0),
                                           (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                           AmoebaMultipoleForce.NoAxisType,
                                           -1, -1, -1,
                                           0.39,  # Thole damping parameter
                                           atom.alpha ** (1 / 6),
                                           atom.alpha
                                           )
                        other_atoms = [atom2.id for atom2 in mol if i != atom2.id]
                        force.setCovalentMap(i, AmoebaMultipoleForce.Covalent12, other_atoms)
                        force.setCovalentMap(i, AmoebaMultipoleForce.PolarizationCovalent11, other_atoms)
        for constraint in self._constraints:
            self._omm_system.addConstraint(*constraint)

    def _set_positions(self):
        positions = np.row_stack([mol.get_positions() for mol in self.mols])
        positions *= nanometers / 10
        self._omm_context.setPositions(positions)

    def create_openmm_context(self):
        self._start_time = time.perf_counter()
        self._omm_system = System()
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
            "c6 = sqrt(c61*c62);"
            "c8 = sqrt(c81*c82);"
            "c10 = sqrt(c101*c102);"
            "invbeta = select(beta, 1.0/beta, 0);"
            "beta = select(beta_mix, 2.0*beta_mix/(beta1+beta2), 0);"
            "beta_mix = beta1*beta2;"
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
        self._omm_system.addForce(tt_force)
        mp_force = AmoebaMultipoleForce()
        mp_force.setNonbondedMethod(AmoebaMultipoleForce.PME)
        mp_force.setPolarizationType(AmoebaMultipoleForce.Extrapolated)
        mp_force.setCutoffDistance(0.9)
        self._omm_system.addForce(mp_force)
        if self.ensemble == GCMCSystem.nvt or self.ensemble == GCMCSystem.npt:
            self._omm_integrator = NoseHooverIntegrator(self.temperature, 1 / picosecond, self.dt)
        elif self.ensemble == GCMCSystem.nve or self.ensemble == GCMCSystem.uvt:
            self._omm_integrator = VerletIntegrator(self.dt)
        else:
            raise Exception("Cannot create integrator for (unknown) ensemble")
        self.add_molecules_to_openmm_system()
        self._omm_system.setDefaultPeriodicBoxVectors(*(self._pbc.basis_matrix * nanometers / 10))
        self._omm_context = Context(self._omm_system, self._omm_integrator)
        self._set_positions()
        self._omm_context.setVelocitiesToTemperature(self.temperature)
        f = open("state.xml", "w")
        f.write(XmlSerializer.serialize(self._omm_system))
        f.close()
        print("OpenMM Context creation time {:5.2f} s".format(time.perf_counter()-self._start_time))

    def create_new_mol(self):
        self.add_sorbate()
        mol = self.mols[-1]
        for atom in mol:
            self._omm_system.addParticle(atom.mass)
        for force in self._omm_system.getForces():
            if isinstance(force, NonbondedForce):
                for atom in mol:
                    force.addParticle(atom.charge, atom.lj_sigma, atom.lj_epsilon)
            elif isinstance(force, CustomNonbondedForce):
                for atom in mol:
                    force.addParticle((atom.c6, atom.c8, atom.c10, atom.beta, atom.rho))
            elif isinstance(force, AmoebaMultipoleForce):
                for atom in mol:
                    force.addMultipole(atom.charge,
                                       (0.0, 0.0, 0.0),
                                       (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                       AmoebaMultipoleForce.NoAxisType,
                                       -1, -1, -1,
                                       0.39,  # Thole damping parameter
                                       atom.alpha ** (1 / 6),
                                       atom.alpha
                                       )
        self._omm_context.reinitialize()
        self._set_positions()

    def output_xyz(self):
        if self._out_file is None:
            self._out_file = open(self.xyz_filename, "w")
        atoms = []
        for mol in self.mols:
            atoms += mol.atoms
        positions = self._omm_state.getPositions(asNumpy=True)
        self._out_file.write("{}\n\n".format(int(len(atoms))))
        for i, atom in enumerate(atoms):
            self._out_file.write("{} {} {} {}\n".format(atom.element, *positions[i].value_in_unit(angstroms)))

    def initial_output(self):
        self._start_time = time.perf_counter()
        print(" --- OpenGCMC ---")
        print("{} ensemble".format(self.ensemble_to_name[self.ensemble]))
        print("Cell basis matrix\n[ {:6.3f} {:6.3f} {:6.3f}\n  "
              "{:6.3f} {:6.3f} {:6.3f}\n  "
              "{:6.3f} {:6.3f} {:6.3f} ]".format(
               self._pbc.basis_matrix[0][0],
               self._pbc.basis_matrix[0][1],
               self._pbc.basis_matrix[0][2],
               self._pbc.basis_matrix[1][0],
               self._pbc.basis_matrix[1][1],
               self._pbc.basis_matrix[1][2],
               self._pbc.basis_matrix[2][0],
               self._pbc.basis_matrix[2][1],
               self._pbc.basis_matrix[2][2],
              ))
        print("dt: {} integrator: {}".format(self.dt, self._omm_integrator.__class__.__name__))
        print("{} atoms in {} molecules".format(self._n, len(self.mols)))
        mol_names = [mol.to_name() for mol in self.mols]
        mol_names, counts = np.unique(mol_names, return_counts=True)
        for i, mol_name in enumerate(mol_names):
            print(" → {}x {}".format(counts[i], mol_name))
        print(" ----------------")

    def output(self):
        current_time = (self._step * self.dt).value_in_unit(nanoseconds)
        if self._step == 0:
            self.initial_output()
            ns_per_day = 0
        else:
            ns_per_day = current_time / (time.perf_counter() - self._start_time) * 86400
        kargs = {"getPositions": False, "getEnergy": True, "enforcePeriodicBox": True}
        if self.write_xyz:
            kargs["getPositions"] = True
        self._omm_state = self._omm_context.getState(**kargs)
        if self.write_xyz:
            self.output_xyz()
        kin_energy = self._omm_state.getKineticEnergy().value_in_unit(kilojoule_per_mole)
        pot_energy = self._omm_state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        nonfrozen_mols = [mol for mol in self.mols if not mol.frozen and not mol.ghost]
        print(self.format_string.format(step=self._step,
                                        time=current_time * 1000,  # ns to ps
                                        tot_energy=kin_energy+pot_energy,
                                        kin_energy=kin_energy,
                                        pot_energy=pot_energy,
                                        elapsed_s=time.perf_counter() - self._start_time,
                                        ns_per_day=ns_per_day,
                                        n=len(nonfrozen_mols)))

    def step(self, steps):
        if self._omm_integrator is None:
            raise Exception("Integrator doesn't exist (add molecules and "
                            "call create_openmm_context() first)")
        if self._step == 0:
            self.output()

        if self.ensemble == GCMCSystem.uvt:
            for _ in range(steps):
                self.hybrid_mc_step(1)
        else:
            self.md_step(steps)

    def md_step(self, steps):

        total_steps = self._step + steps
        while self._step < total_steps:
            if self._step + self.freq <= total_steps:
                delta_steps = self.freq
            else:
                delta_steps = total_steps - self._step
            self._omm_integrator.step(delta_steps)
            self._step += delta_steps
            self.output()

    def hybrid_mc_step(self, steps, md_steps=100):

        total_steps = self._step + steps
        while self._step < total_steps:

            self._omm_state = self._omm_context.getState(getEnergy=True, getPositions=True)
            old_pot_energy = self._omm_state.getPotentialEnergy()
            old_positions = self._omm_state.getPositions(asNumpy=True)
            self._omm_context.setVelocitiesToTemperature(self.temperature)
            self._omm_integrator.step(md_steps)
            new_state = self._omm_context.getState(getEnergy=True)
            new_pot_energy = new_state.getPotentialEnergy()
            delta_e = new_pot_energy.value_in_unit(kilojoule_per_mole) - \
                      old_pot_energy.value_in_unit(kilojoule_per_mole)
            boltzmann_factor = np.exp(-delta_e/(self.temperature.value_in_unit(kelvins) * 0.008314462618))

            if random.random() > boltzmann_factor:
                self._omm_context.setPositions(old_positions)

            self._step += 1
            if self._step % self.freq == 0:
                self.output()

    '''
        def unghost_mol(self, mol: Molecule):
        mol.ghost = False
        for force in self._omm_system.getForces():
            if isinstance(force, NonbondedForce):
                for atom in mol:
                    force.setParticleParameters(atom.id, atom.charge, atom.lj_sigma, atom.lj_epsilon)
            elif isinstance(force, CustomNonbondedForce):
                for atom in mol:
                    force.setParticleParameters(atom.id, (atom.c6, atom.c8, atom.c10, atom.beta, atom.rho))
            elif isinstance(force, AmoebaMultipoleForce):
                for atom in mol:
                    force.setMultipoleParameters(atom.id, atom.charge,
                                                 (0.0, 0.0, 0.0),
                                                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                                 AmoebaMultipoleForce.NoAxisType,
                                                 -1, -1, -1,
                                                 0.39,  # Thole damping parameter
                                                 atom.alpha ** (1 / 6),
                                                 atom.alpha
                                                 )
            force.updateParametersInContext(self._omm_context)
        for atom in mol:
            self._omm_system.setParticleMass(atom.id, atom.mass)


    def ghost_mol(self, mol: Molecule):
        mol.ghost = True
        for force in self._omm_system.getForces():
            if isinstance(force, NonbondedForce):
                for atom in mol:
                    force.setParticleParameters(atom.id, 0.0, 0.0, 0.0)
            elif isinstance(force, CustomNonbondedForce):
                for atom in mol:
                    force.setParticleParameters(atom.id, (0.0, 0.0, 0.0, 0.0, 0.0))
            elif isinstance(force, AmoebaMultipoleForce):
                for atom in mol:
                    force.setMultipoleParameters(atom.id, 0.0,
                                                 (0.0, 0.0, 0.0),
                                                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                                 AmoebaMultipoleForce.NoAxisType,
                                                 -1, -1, -1,
                                                 0.39,  # Thole damping parameter
                                                 0.0,
                                                 0.0
                                                 )
            force.updateParametersInContext(self._omm_context)
        for atom in mol:
            self._omm_system.setParticleMass(atom.id, 0.0)
    
    
    def muvt_steps(self, steps):

        total_steps = self._step + steps
        while self._step < total_steps:
            self._omm_state = self._omm_context.getState(getEnergy=True, getPositions=True)
            old_pot_energy = self._omm_state.getPotentialEnergy()
            old_positions = self._omm_state.getPositions(asNumpy=True)

            if np.random.random() > 0.5:
                # delete
                mol_indices = [i for i, mol in enumerate(self.mols) if not mol.frozen and not mol.ghost]
                if len(mol_indices) == 0:
                    continue
                chosen_mol = np.random.choice(mol_indices)
                self.ghost_mol(self.mols[chosen_mol])
                new_state = self._omm_context.getState(getEnergy=True)
                new_pot_energy = new_state.getPotentialEnergy()
                delta_e = new_pot_energy.value_in_unit(kilojoule_per_mole) - \
                          old_pot_energy.value_in_unit(kilojoule_per_mole)
                boltzmann_factor = self.temperature.value_in_unit(kelvins) * len(mol_indices) \
                                   / (self._pbc.volume * self.pressure.value_in_unit(atmospheres) * 0.0073389366) *\
                                   np.exp(-delta_e / (self.temperature.value_in_unit(kelvins) * 0.008314462618))
                #print("deleting mol {} before {} after {} bf {}".format(chosen_mol, old_pot_energy, new_pot_energy, boltzmann_factor))
                if random.random() > boltzmann_factor:
                    self.unghost_mol(self.mols[chosen_mol])
                    #print("undo deleting")
                else:
                    self._omm_context.reinitialize()
                    self._omm_context.setPositions(old_positions)
                    self._omm_context.setVelocitiesToTemperature(self.temperature)

            else:
                # insert
                mol_indices = [i for i, mol in enumerate(self.mols) if not mol.frozen and mol.ghost]
                if len(mol_indices) == 0:
                    self.create_new_mol()
                    chosen_mol = len(self.mols) - 1
                    self._omm_context.reinitialize()
                else:
                    chosen_mol = np.random.choice(mol_indices)
                    self.unghost_mol(self.mols[chosen_mol])
                Modeler.move_mol_randomly(self.mols[chosen_mol], self._pbc)
                positions = np.row_stack([mol.get_positions() for mol in self.mols])
                positions *= nanometers / 10
                self._omm_context.setPositions(positions)
                new_state = self._omm_context.getState(getEnergy=True)
                new_pot_energy = new_state.getPotentialEnergy()
                delta_e = new_pot_energy.value_in_unit(kilojoule_per_mole) - \
                          old_pot_energy.value_in_unit(kilojoule_per_mole)
                if delta_e < 1e4:
                    boltzmann_factor = (self._pbc.volume * self.pressure.value_in_unit(atmospheres) * 0.0073389366) / \
                                       (self.temperature.value_in_unit(kelvins) * (len(mol_indices)+1)) * \
                                        np.exp(-delta_e / (self.temperature.value_in_unit(kelvins) * 0.008314462618))
                else:
                    boltzmann_factor = 0.0
                #print("inserting mol {} before {} after {} bf {}".format(chosen_mol, old_pot_energy, new_pot_energy, boltzmann_factor))

                if np.random.random() > boltzmann_factor:
                    #print("undo insert")
                    self.ghost_mol(self.mols[chosen_mol])
                else:
                    pass


            self._step += 1
            if self._step % self.freq == 0:
                self.output()
    '''
