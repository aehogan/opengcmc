

class PhahstFF:
    phahst = {
        "Cu": {
            "mass": 63.54630,
            "alpha": 0.29252,
            "sigma": 2.73851,
            "epsilon": 8.82345,
            "c6": 6.96956,
            "c8": 262.82938,
            "c10": 13951.49740,
        },
        "C": {
            "mass": 12.01100,
            "alpha": 0.71317,
            "sigma": 3.35929,
            "epsilon": 4.00147,
            "c6": 11.88969,
            "c8": 547.51694,
            "c10": 27317.97855,
        },
        "O": {
            "mass": 15.99900,
            "alpha": 1.68064,
            "sigma": 3.23867,
            "epsilon": 3.89544,
            "c6": 27.70093,
            "c8": 709.36452,
            "c10": 19820.89339,
        },
        "H": {
            "mass": 1.00790,
            "alpha": 0.02117,
            "sigma": 1.87446,
            "epsilon": 3.63874,
            "c6": 0.16278,
            "c8": 5.03239,
            "c10": 202.99322,
        },
    }

    phahst_h2 = {
        "H": {
            "mass": 1.00790,
            "alpha": 0.34325,
            "sigma": 1.859425,
            "epsilon": 3.100603,
            "c6": 2.884735,
            "c8": 38.97178,
            "c10": 644.95683,
        },
        "Da": {
            "mass": 0.0,
            "alpha": 0.0,
            "sigma": 2.664506,
            "epsilon": 3.627796,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        }
    }

    def __init__(self):
        pass

    @staticmethod
    def apply(atoms, ff):
        for atom in atoms:
            try:
                atom.mass = ff[atom.element]["mass"]
                atom.alpha = ff[atom.element]["alpha"]
                atom.epsilon = ff[atom.element]["epsilon"]
                atom.sigma = ff[atom.element]["sigma"]
                atom.c6 = ff[atom.element]["c6"]
                atom.c8 = ff[atom.element]["c8"]
                atom.c10 = ff[atom.element]["c10"]
            except KeyError:
                Exception("atom {} not found in forcefield".format(atom.element))
