import numpy as np


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
        self.mass = np.sum([atom.mass for atom in self.atoms])

    def update_mass(self):
        self.mass = np.sum([atom.mass for atom in self.atoms])

    def append(self, atom):
        self.atoms.append(atom)
        self.update_mass()

    def __getitem__(self, i):
        return self.atoms[i]

    def __setitem__(self, key, value):
        self.atoms[key] = value
        self.update_mass()

    def __len__(self):
        return len(self.atoms)

    def __iter__(self):
        yield from self.atoms

    def get_positions(self):
        return np.row_stack([atom.x for atom in self.atoms])

    def to_name(self):
        elements, counts = np.unique([atom.element for atom in self.atoms], return_counts=True)
        name = ""
        for i, element in enumerate(elements):
            name += element + str(counts[i])
            if i != len(elements) - 1:
                name += " "
        return name
