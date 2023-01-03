
from .Atom import Atom, Molecule
from openmm import TwoParticleAverageSite


class Sorbate:
    def __init__(self, base_id):
        self.molecule = Molecule()
        self.constraints = []
        self.base_id = base_id


class H2(Sorbate):
    def __init__(self, base_id):
        super().__init__(base_id)
        self.molecule.append(Atom(0.0, 0.0, 0.0, "DAH2", atom_id=base_id, charge=-0.846166, virtual=True,
                             virtual_type=TwoParticleAverageSite(base_id + 1, base_id + 2, 0.5, 0.5)))
        self.molecule.append(Atom(0.371, 0.0, 0.0, "H2", atom_id=base_id + 1, charge=0.423083))
        self.molecule.append(Atom(-0.371, 0.0, 0.0, "H2", atom_id=base_id + 2, charge=0.423083))
        PhahstFF.apply_ff(self.molecule, PhahstFF.phahst_h2)
        self.constraints.append([base_id + 1, base_id + 2, 2 * 0.371 / 10])


class PhahstFF:
    phahst = {
        "Cu": {
            "mass": 63.54630,
            "alpha": 0.29252 / 1000,
            "rho": 2.73851 / 10,
            "beta": 8.82345 * 10,
            "c6": 6.96956 * 2625.5 / 18.8973**6,
            "c8": 262.82938 * 2625.5 / 18.8973**8,
            "c10": 13951.49740 * 2625.5 / 18.8973**10,
        },
        "C": {
            "mass": 12.01100,
            "alpha": 0.71317 / 1000,
            "rho": 3.35929 / 10,
            "beta": 4.00147 * 10,
            "c6": 11.88969 * 2625.5 / 18.8973**6,
            "c8": 547.51694 * 2625.5 / 18.8973**8,
            "c10": 27317.97855 * 2625.5 / 18.8973**10,
        },
        "O": {
            "mass": 15.99900,
            "alpha": 1.68064 / 1000,
            "rho": 3.23867 / 10,
            "beta": 3.89544 * 10,
            "c6": 27.70093 * 2625.5 / 18.8973**6,
            "c8": 709.36452 * 2625.5 / 18.8973**8,
            "c10": 19820.89339 * 2625.5 / 18.8973**10,
        },
        "H": {
            "mass": 1.00790,
            "alpha": 0.02117 / 1000,
            "rho": 1.87446 / 10,
            "beta": 3.63874 * 10,
            "c6": 0.16278 * 2625.5 / 18.8973**6,
            "c8": 5.03239 * 2625.5 / 18.8973**8,
            "c10": 202.99322 * 2625.5 / 18.8973**10,
        },
    }

    phahst_h2 = {
        "H": {
            "mass": 1.00790,
            "alpha": 0.34325 / 1000,
            "rho": 1.859425 / 10,
            "beta": 3.100603 * 10,
            "c6": 2.884735 * 2625.5 / 18.8973**6,
            "c8": 38.97178 * 2625.5 / 18.8973**8,
            "c10": 644.95683 * 2625.5 / 18.8973**10,
        },
        "Da": {
            "mass": 0.0,
            "alpha": 0.0,
            "rho": 2.664506 / 10,
            "beta": 3.627796 * 10,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        }
    }

    def __init__(self):
        pass

    @staticmethod
    def apply_ff(atoms, ff):
        for atom in atoms:
            try:
                atom.mass = ff[atom.element]["mass"]
                atom.alpha = ff[atom.element]["alpha"]
                atom.beta = ff[atom.element]["beta"]
                atom.rho = ff[atom.element]["rho"]
                atom.c6 = ff[atom.element]["c6"]
                atom.c8 = ff[atom.element]["c8"]
                atom.c10 = ff[atom.element]["c10"]
            except KeyError:
                Exception("atom {} not found in forcefield".format(atom.element))
