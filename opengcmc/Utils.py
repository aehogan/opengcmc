import numpy as np
import copy


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
        self.x = np.array([x, y, z, w], dtype=float)

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


class Modeler:
    def __init__(self):
        pass

    @staticmethod
    def overlap_mol_test(input_mol, mol_check_list, pbc):
        for atom1 in input_mol:
            for mol2 in mol_check_list:
                for atom2 in mol2:
                    r = pbc.min_image(atom1.x - atom2.x)
                    if r < 3:
                        return True
        return False

    @staticmethod
    def move_mol_randomly(input_mol, pbc):
        Modeler.move_mol_to_com(input_mol)
        random_rot = Quaternion()
        random_rot.random_rotation()
        random_scale = np.random.random(3)
        random_scale = np.matmul(random_scale, pbc.basis_matrix)
        for atom in input_mol:
            atom.x = Quaternion.rotate_3vector(atom.x, random_rot)
            atom.x += random_scale

    @staticmethod
    def move_mol_to_com(atoms):
        com = np.zeros(3)
        total_mass = 0
        for atom in atoms:
            com += atom.x * atom.mass
            total_mass += atom.mass
        com /= total_mass
        for atom in atoms:
            atom.x -= com

    @staticmethod
    def point_molecule_down_xaxis(atoms, atom_index):

        pointer_atom = copy.deepcopy(atoms[atom_index])

        xangle = np.arctan2(pointer_atom.x[1], pointer_atom.x[0])
        qx = Quaternion(0.0, 0.0, 0.0, 1.0)
        qx.axis_angle(0, 0, 1, -np.rad2deg(xangle))

        pointer_atom.x = Quaternion.rotate_3vector(pointer_atom.x, qx)

        yangle = np.arctan2(pointer_atom.x[2], pointer_atom.x[0])
        qy = Quaternion(0.0, 0.0, 0.0, 1.0)
        qy.axis_angle(0, 1, 0, np.rad2deg(yangle))

        pointer_atom.x = Quaternion.rotate_3vector(pointer_atom.x, qy)

        for atom in atoms:
            atom.x = Quaternion.rotate_3vector(atom.x, qx)
            atom.x = Quaternion.rotate_3vector(atom.x, qy)
            atom.x -= pointer_atom.x

    @staticmethod
    def invert_mol(atoms):
        for atom in atoms:
            atom.x = -atom.x
