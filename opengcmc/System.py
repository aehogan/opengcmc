import numpy as np
import copy
from opengcmc import Quaternion, PBC, PhahstFF
from openmm import NonbondedForce, CustomNonbondedForce, AmoebaMultipoleForce,\
    System, NoseHooverIntegrator, Context, TwoParticleAverageSite, XmlSerializer
from openmm.unit import *


class Atom:
    list_of_elements = [
        "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk", "Br", "C",
        "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Co", "Cr", "Cs", "Cu", "Db", "Dy", "Er", "Es", "Eu", "F",
        "Fe", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg", "Ho", "Hs", "I", "In", "Ir", "K",
        "Kr", "La", "Li", "Lr", "Lu", "Md", "Mg", "Mn", "Mo", "Mt", "N", "Na", "Nb", "Nd", "Ne", "Ni",
        "No", "Np", "O", "Os", "P", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re",
        "Rf", "Rh", "Rn", "Ru", "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc",
        "Te", "Th", "Ti", "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr", "Da",
    ]

    element_masses = {
        "H": 1.00797, "He": 4.0026, "Li": 6.941, "Be": 9.01218, "B": 10.81, "C": 12.011, "N": 14.0067,
        "O": 15.9994, "F": 18.998403, "Ne": 20.179, "Na": 22.98977, "Mg": 24.305, "Al": 26.98154,
        "Si": 28.0855, "P": 30.97376, "S": 32.06, "Cl": 35.453, "K": 39.0983, "Ar": 39.948,
        "Ca": 40.08, "Sc": 44.9559, "Ti": 47.9, "V": 50.9415, "Cr": 51.996, "Mn": 54.938, "Fe": 55.847,
        "Ni": 58.7, "Co": 58.9332, "Cu": 63.546, "Zn": 65.38, "Ga": 69.72, "Ge": 72.59, "As": 74.9216,
        "Se": 78.96, "Br": 79.904, "Kr": 83.8, "Rb": 85.4678, "Sr": 87.62, "Y": 88.9059, "Zr": 91.22,
        "Nb": 92.9064, "Mo": 95.94, "Tc": 98, "Ru": 101.07, "Rh": 102.9055, "Pd": 106.4, "Ag": 107.868,
        "Cd": 112.41, "In": 114.82, "Sn": 118.69, "Sb": 121.75, "I": 126.9045, "Te": 127.6,
        "Xe": 131.3, "Cs": 132.9054, "Ba": 137.33, "La": 138.9055, "Ce": 140.12, "Pr": 140.9077,
        "Nd": 144.24, "Pm": 145, "Sm": 150.4, "Eu": 151.96, "Gd": 157.25, "Tb": 158.9254, "Dy": 162.5,
        "Ho": 164.9304, "Er": 167.26, "Tm": 168.9342, "Yb": 173.04, "Lu": 174.967, "Hf": 178.49,
        "Ta": 180.9479, "W": 183.85, "Re": 186.207, "Os": 190.2, "Ir": 192.22, "Pt": 195.09,
        "Au": 196.9665, "Hg": 200.59, "Tl": 204.37, "Pb": 207.2, "Bi": 208.9804, "Po": 209, "At": 210,
        "Rn": 222, "Fr": 223, "Ra": 226.0254, "Ac": 227.0278, "Pa": 231.0359, "Th": 232.0381,
        "Np": 237.0482, "U": 238.029, "Pu": 242, "Am": 243, "Bk": 247, "Cm": 247, "No": 250, "Cf": 251,
        "Es": 252, "Hs": 255, "Mt": 256, "Fm": 257, "Md": 258, "Lr": 260, "Rf": 261, "Bh": 262,
        "Db": 262, "Sg": 263, "Da": 0,
    }

    element_z = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
        "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
        "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
        "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
        "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
        "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
        "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
        "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
        "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
        "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
        "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108,
        "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116,
        "Ts": 117, "Og": 118, "Da": 0,
    }

    def __init__(self, x, y, z, name, atom_id=0, charge=0.0, virtual=False, virtual_type=None):
        self.name = name.strip()
        self.x = np.array([float(x), float(y), float(z)])
        element = "".join([i for i in self.name[:2] if i.isalpha()])
        element = element.lower().capitalize()

        if element not in self.list_of_elements:
            element = element[0]
            if element not in self.list_of_elements:
                raise Exception("Couldn't find element {}".format(name))

        self.virtual = virtual
        self.virtual_type = virtual_type
        self.element = element
        self.charge = charge
        self.alpha = 0.0
        self.lj_epsilon = 0.0
        self.lj_sigma = 0.0
        self.beta = 0.0
        self.rho = 0.0
        self.c6 = 0.0
        self.c8 = 0.0
        self.c10 = 0.0
        self.mass = self.element_masses[self.element]
        self.z = self.element_z[self.element]
        self.id = atom_id


class Molecule:
    def __init__(self, name="mol", atoms=None, frozen=False):
        if atoms is None:
            atoms = []
        self.name = str(name)
        self.atoms = atoms
        self.frozen = frozen

    def append(self, atom):
        self.atoms.append(atom)

    def __getitem__(self, i):
        return self.atoms[i]

    def __setitem__(self, key, value):
        self.atoms[key] = value

    def __len__(self):
        return len(self.atoms)

    def __iter__(self):
        yield from self.atoms

    def get_positions(self):
        return np.row_stack([atom.x for atom in self.atoms])

    def move_to_com(self):
        com = np.zeros(3)
        total_mass = 0
        for atom in self.atoms:
            com += atom.x * atom.mass
            total_mass += atom.mass
        com /= total_mass
        for atom in self.atoms:
            atom.x -= com

    def point_molecule_down_xaxis(self, atom_index):

        pointer_atom = copy.deepcopy(self.atoms[atom_index])

        xangle = np.arctan2(pointer_atom.x[1], pointer_atom.x[0])
        qx = Quaternion(0.0, 0.0, 0.0, 1.0)
        qx.axis_angle(0, 0, 1, -np.rad2deg(xangle))

        pointer_atom.x = Quaternion.rotate_3vector(pointer_atom.x, qx)

        yangle = np.arctan2(pointer_atom.x[2], pointer_atom.x[0])
        qy = Quaternion(0.0, 0.0, 0.0, 1.0)
        qy.axis_angle(0, 1, 0, np.rad2deg(yangle))

        pointer_atom.x = Quaternion.rotate_3vector(pointer_atom.x, qy)

        for atom in self.atoms:
            atom.x = Quaternion.rotate_3vector(atom.x, qx)
            atom.x = Quaternion.rotate_3vector(atom.x, qy)
            atom.x -= pointer_atom.x

    def invert(self):
        for atom in self.atoms:
            atom.x = -atom.x


class GCMCSystem:
    # Thermodynamic ensemble enum
    uvt, npt, nvt, nve = range(4)

    def __init__(self):
        self.ensemble = GCMCSystem.nvt
        self.pressure = 1.0 * atmospheres
        self.temperature = 298.0 * kelvins
        # Frequency of output
        self.freq = 10
        # Output f-string
        self.format_string = "Step {step} KinE {kin_energy} PotE {pot_energy}"
        # Write .xyz flag/filename
        self.write_xyz = True
        self.xyz_filename = "out.xyz"
        # list of Molecule objects
        self.mols = []
        # Current number of atoms
        self._n = 0
        # Current step number
        self._step = 0

        # internal objects
        self._pbc = None
        self._out_file = None
        self._ff = PhahstFF()

        # OpenMM objects
        self._omm_system = None
        self._omm_integrator = None
        self._omm_context = None
        self._omm_state = None
        self._constraints = []

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
            self._ff.apply(mof.atoms, self._ff.phahst)
            self.mols.append(mof)

    def add_sorbate(self, sorbate):
        if sorbate == "H2":
            h2 = Molecule()
            h2.append(Atom(0.0, 0.0, 0.0, "DAH2", atom_id=self._n, charge=-0.846166, virtual=True,
                           virtual_type=TwoParticleAverageSite(self._n+1, self._n+2, 0.5, 0.5)))
            h2.append(Atom(0.0371, 0.0, 0.0, "H2", atom_id=self._n+1, charge=0.423083))
            h2.append(Atom(-0.0371, 0.0, 0.0, "H2", atom_id=self._n+2, charge=0.423083))
            self._constraints.append([self._n+1, self._n+2, 2*0.0371])
            self._ff.apply(h2, self._ff.phahst_h2)
            self._n += 3
            self.mols.append(h2)
        else:
            raise Exception("Unknown sorbate")

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

    def create_openmm_context(self):
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
        tt_force.setCutoffDistance(1.2)
        tt_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        tt_force.setUseLongRangeCorrection(False)
        self._omm_system.addForce(tt_force)
        mp_force = AmoebaMultipoleForce()
        mp_force.setNonbondedMethod(AmoebaMultipoleForce.PME)
        mp_force.setPolarizationType(AmoebaMultipoleForce.Extrapolated)
        mp_force.setCutoffDistance(1.2)
        # self._omm_system.addForce(mp_force)
        self._omm_integrator = NoseHooverIntegrator(self.temperature, 1 / picosecond, 0.001 * picoseconds)
        self.add_molecules_to_openmm_system()
        self._omm_system.setDefaultPeriodicBoxVectors(*(self._pbc.basis_matrix * nanometers / 10))
        self._omm_context = Context(self._omm_system, self._omm_integrator)
        positions = np.row_stack([mol.get_positions() for mol in self.mols])
        self._omm_context.setPositions(positions)
        self._omm_context.setVelocitiesToTemperature(298)
        f = open("state.xml", "w")
        f.write(XmlSerializer.serialize(self._omm_system))
        f.close()

    def output_xyz(self):
        if self._out_file is None:
            self._out_file = open(self.xyz_filename, "w")
        atoms = []
        for mol in self.mols:
            atoms += mol.atoms
        positions = self._omm_state.getPositions(asNumpy=True)
        self._out_file.write("{}\n\n".format(int(len(atoms))))
        for i, atom in enumerate(atoms):
            self._out_file.write("{} {} {} {}\n".format(atom.element, *positions[i]._value*10))

    def output(self):
        params = {"getPositions": False, "getEnergy": True, "enforcePeriodicBox": True}
        if self.write_xyz:
            params["getPositions"] = True
        self._omm_state = self._omm_context.getState(**params)
        if self.write_xyz:
            self.output_xyz()
        print(self.format_string.format(step=self._step,
                                        kin_energy=self._omm_state.getKineticEnergy(),
                                        pot_energy=self._omm_state.getPotentialEnergy()))

    def step(self, steps):
        if self._omm_integrator is None:
            raise Exception("Integrator doesn't exist (add molecules and"
                            "call create_openmm_context() first)")
        if self._step == 0:
            self.output()
        total_steps = self._step + steps
        while self._step < total_steps:
            if self._step + self.freq <= total_steps:
                delta_steps = self.freq
            else:
                delta_steps = total_steps - self._step
            self._omm_integrator.step(delta_steps)
            self._step += delta_steps
            self.output()
