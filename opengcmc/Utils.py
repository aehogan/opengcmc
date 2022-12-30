import numpy as np
import copy


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

    def __init__(self, x, y, z, name, atom_id=0, charge=0.0):
        self.name = name.strip()
        self.x = np.array([float(x), float(y), float(z)])
        element = "".join([i for i in self.name[:2] if i.isalpha()])
        element = element.lower().capitalize()

        if element not in self.list_of_elements:
            element = element[0]
            if element not in self.list_of_elements:
                print("!!! Invalid element {} !!!".format(name))

        if element == "H":
            self.bond_r = 0.8
            self.vdw = 1.2
        elif element == "O":
            self.bond_r = 1.3
            self.vdw = 1.8
        elif element == "N" or element == "C":
            self.bond_r = 1.6
            self.vdw = 2.0
        else:
            self.bond_r = 2.0
            self.vdw = 3.0

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

    def __str__(self):
        return "Atom {} {} [{}]".format(self.element, self.id, self.x)


class Molecule:
    def __init__(self, name="mol", atoms=None, charge=0, mult=1):
        if atoms is None:
            atoms = []
        self.name = str(name)
        self.atoms = atoms
        self.charge = int(charge)
        self.mult = int(mult)

    def append(self, atom):
        self.atoms.append(atom)

    def __getitem__(self, item):
        return self.atoms[item]

    def __setitem__(self, key, value):
        self.atoms[key] = value

    def __len__(self):
        return len(self.atoms)

    def __iter__(self):
        yield from self.atoms

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


class PBC:
    def __init__(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        basis00 = a
        basis01 = 0.0
        basis02 = 0.0
        basis10 = b * np.cos(np.pi / 180.0 * gamma)
        basis11 = b * np.sin(np.pi / 180.0 * gamma)
        basis12 = 0.0
        basis20 = c * np.cos(np.pi / 180.0 * beta)
        basis21 = ((b * c * np.cos(np.pi / 180.0 * alpha)) - (basis10 * basis20)) / basis11
        basis22 = np.sqrt(c * c - basis20 * basis20 - basis21 * basis21)

        self.basis_matrix = np.array(
            [
                [basis00, basis01, basis02],
                [basis10, basis11, basis12],
                [basis20, basis21, basis22],
            ]
        )

        self.volume = basis00 * (basis11 * basis22 - basis12 * basis21)
        self.volume += basis01 * (basis12 * basis20 - basis10 * basis22)
        self.volume += basis02 * (basis10 * basis21 - basis11 * basis20)

        self.inverse_volume = 1.0 / self.volume

        reciprocal_basis00 = self.inverse_volume * (basis11 * basis22 - basis12 * basis21)
        reciprocal_basis01 = self.inverse_volume * (basis02 * basis21 - basis01 * basis22)
        reciprocal_basis02 = self.inverse_volume * (basis01 * basis12 - basis02 * basis11)
        reciprocal_basis10 = self.inverse_volume * (basis12 * basis20 - basis10 * basis22)
        reciprocal_basis11 = self.inverse_volume * (basis00 * basis22 - basis02 * basis20)
        reciprocal_basis12 = self.inverse_volume * (basis02 * basis10 - basis00 * basis12)
        reciprocal_basis20 = self.inverse_volume * (basis10 * basis21 - basis11 * basis20)
        reciprocal_basis21 = self.inverse_volume * (basis01 * basis20 - basis00 * basis21)
        reciprocal_basis22 = self.inverse_volume * (basis00 * basis11 - basis01 * basis10)

        self.reciprocal_basis_matrix = np.array(
            [
                [reciprocal_basis00, reciprocal_basis01, reciprocal_basis02],
                [reciprocal_basis10, reciprocal_basis11, reciprocal_basis12],
                [reciprocal_basis20, reciprocal_basis21, reciprocal_basis22],
            ]
        )

    def update(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        basis00 = a
        basis01 = 0.0
        basis02 = 0.0
        basis10 = b * np.cos(np.pi / 180.0 * gamma)
        basis11 = b * np.sin(np.pi / 180.0 * gamma)
        basis12 = 0.0
        basis20 = c * np.cos(np.pi / 180.0 * beta)
        basis21 = ((b * c * np.cos(np.pi / 180.0 * alpha)) - (basis10 * basis20)) / basis11
        basis22 = np.sqrt(c * c - basis20 * basis20 - basis21 * basis21)

        self.basis_matrix = np.array(
            [
                [basis00, basis01, basis02],
                [basis10, basis11, basis12],
                [basis20, basis21, basis22],
            ]
        )

        self.volume = basis00 * (basis11 * basis22 - basis12 * basis21)
        self.volume += basis01 * (basis12 * basis20 - basis10 * basis22)
        self.volume += basis02 * (basis10 * basis21 - basis11 * basis20)

        self.inverse_volume = 1.0 / self.volume

        reciprocal_basis00 = self.inverse_volume * (basis11 * basis22 - basis12 * basis21)
        reciprocal_basis01 = self.inverse_volume * (basis02 * basis21 - basis01 * basis22)
        reciprocal_basis02 = self.inverse_volume * (basis01 * basis12 - basis02 * basis11)
        reciprocal_basis10 = self.inverse_volume * (basis12 * basis20 - basis10 * basis22)
        reciprocal_basis11 = self.inverse_volume * (basis00 * basis22 - basis02 * basis20)
        reciprocal_basis12 = self.inverse_volume * (basis02 * basis10 - basis00 * basis12)
        reciprocal_basis20 = self.inverse_volume * (basis10 * basis21 - basis11 * basis20)
        reciprocal_basis21 = self.inverse_volume * (basis01 * basis20 - basis00 * basis21)
        reciprocal_basis22 = self.inverse_volume * (basis00 * basis11 - basis01 * basis10)

        self.reciprocal_basis_matrix = np.array(
            [
                [reciprocal_basis00, reciprocal_basis01, reciprocal_basis02],
                [reciprocal_basis10, reciprocal_basis11, reciprocal_basis12],
                [reciprocal_basis20, reciprocal_basis21, reciprocal_basis22],
            ]
        )

    def min_image(self, dx):
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.round(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        r = np.sqrt(np.dot(dx_return, dx_return))
        return r

    def wrap(self, dx):
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.round(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        return dx_return

    def wrap_forward(self, dx):
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.floor(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        return dx_return


class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = np.array([x, y, z, w])

    def normalize(self):
        magnitude = np.linalg.norm(self.x)
        self.x = self.x / magnitude

    def get_conjugate(self):
        result = Quaternion(-self.x[0], -self.x[1], -self.x[2], self.x[3])
        return result

    def axis_angle(self, x, y, z, degrees):
        angle = degrees / 57.2957795
        magnitude = np.linalg.norm(np.array([x, y, z]))
        self.x[0] = x * np.sin(angle / 2.0) / magnitude
        self.x[1] = y * np.sin(angle / 2.0) / magnitude
        self.x[2] = z * np.sin(angle / 2.0) / magnitude
        self.x[3] = np.cos(angle / 2.0)

    def random_rotation(self):
        self.x[0] = np.random.random() * 2.0 - 1.0
        _sum = self.x[0] * self.x[0]
        self.x[1] = np.sqrt(1.0 - _sum) * (np.random.random() * 2.0 - 1.0)
        _sum += self.x[1] * self.x[1]
        self.x[2] = np.sqrt(1.0 - _sum) * (np.random.random() * 2.0 - 1.0)
        _sum += self.x[2] * self.x[2]
        self.x[3] = np.sqrt(1.0 - _sum) * (-1.0 if np.random.random() > 0.5 else 1.0)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            x = self.x[3] * other.x[0] + other.x[3] * self.x[0] + self.x[1] * other.x[2] - self.x[2] * other.x[1]
            y = self.x[3] * other.x[1] + other.x[3] * self.x[1] + self.x[2] * other.x[0] - self.x[0] * other.x[2]
            z = self.x[3] * other.x[2] + other.x[3] * self.x[2] + self.x[0] * other.x[1] - self.x[1] * other.x[0]
            w = self.x[3] * other.x[3] - self.x[0] * other.x[0] - self.x[1] * other.x[1] - self.x[2] * other.x[2]
            result = Quaternion(x, y, z, w)
            return result
        elif isinstance(other, int) or isinstance(other, float):
            result = Quaternion(other * self.x[0], other * self.x[1], other * self.x[2], other * self.x[3])
            return result
        else:
            raise TypeError

    @staticmethod
    def rotate_3vector(x, q):
        vec = Quaternion(x[0], x[1], x[2], 0.)

        result = vec * q.get_conjugate()
        result = q * result
        return result.x[:3]
