import os
import logging
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np
import math
from rdkit.Chem import GetPeriodicTable
from rdkit import Chem
import warnings
from multiprocessing import Pool, freeze_support


def multi_screening(args):
    region_file, fragment_file, output_dir, alignment = args
    r = TWN_FS(region_file, fragment_file, alignment)
    output_file = os.path.join(output_dir, os.path.basename(fragment_file))
    r.to_sdf(output_file)
    return

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TWN_FS:

    def __init__(self, region_file, fragment_file, alignment):
        self.path = os.path.dirname(region_file)
        with open(region_file, 'r') as f:
            lines = f.readlines()
            naming = True
            coordinates = []
            for idx, line in enumerate(lines):
                if naming:
                    self.name = line.split()
                    naming = False
                if len(line) == 40 and len(line.split()) == 6:
                    coordinates += [(float(line.split()[0]), float(line.split()[1]), float(line.split()[2]))]
                if line.startswith('> <region.frequency>'):
                    self.frequency = int(lines[idx + 1].strip())
                if line.startswith('> <twn.pattern.names>'):
                    self.patterns = lines[idx + 1].strip().split()
            self.focus = tuple(np.mean(coordinates, axis=0))
        self.pattern_coordinates, self.pattern_density = self.get_include_patterns()
        self.pattern_pharma_types, self.pattern_pharma_coords, self.pattern_pharma_vectors = self.get_pharmacophores()
        self.output_file = self.fragment_screening(fragment_file, alignment)


    def __call__(self, *args, **kwargs):
        return print(f'<TWN-region object>\nFile path: {self.path}\nID: {self.name}\nFrequency: {self.frequency}\nNumber of patterns included: {len(self.patterns)}')

    # pattern 정보 모으기
    def get_include_patterns(self):
        path = self.path.replace("TWN-Region", "TWN-Pattern")
        patterns_dict = {}
        pattern_density = {}
        with open(os.path.join(path, "TWN.sdf")) as f:
            lines = f.readlines()
            naming = True
            coordinates = []
            for idx, line in enumerate(lines):
                if naming:
                    name = line.strip()
                    naming = False
                if len(line) == 40 and len(line.split()) == 6 and name in self.patterns:
                    coordinates += [(float(line.split()[0]), float(line.split()[1]), float(line.split()[2]))]
                if line.startswith('> <twn.frequency>'):
                    density = round(int(lines[idx + 1].strip())/1000, 3)
                if line.startswith('$$$$'):
                    if len(coordinates) != 0:
                        patterns_dict[name] = coordinates
                        pattern_density[name] = density
                    naming = True
                    coordinates = []
        return patterns_dict, pattern_density

    def get_pharmacophores(self):
        path = self.path.replace("TWN-Region", "TWN-pharma")
        patterns_pharma_types = {}
        patterns_pharma_coords = {}
        patterns_pharma_vectors = {}
        with open(path + "/Pharmacophore.phm") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('TWN_Pattern'):
                    name = line.strip()
                    patterns_pharma_types[name] = []
                    patterns_pharma_coords[name] = []
                    patterns_pharma_vectors[name] = []
                    t, c, v = [], [], []
                if line.startswith('FEAT') and name in self.patterns:
                    t += [line.split()[5]]
                    c += [[float(x) for x in line.split()[2:5]]]
                    v += [[float(x) for x in line.split()[7:10]]]
                if line.startswith('$$$$') and name in self.patterns:
                    center_coord = c[t.index("AR")]
                    for idx, coord in enumerate(c):
                        if t[idx] == "AR":
                            continue # remove AR feature
                            # patterns_pharma_types[name] += [t[idx]]
                        else:
                            compare_vector = np.array(coord) - np.array(center_coord)
                            # Hydrogen bond donor
                            if np.dot(-np.array(v[idx]), compare_vector) > 0:
                                patterns_pharma_types[name] += ["HBD"]
                            # Hydrogen bond acceptor
                            else:
                                patterns_pharma_types[name] += ["HBA"]
                        patterns_pharma_coords[name] += [coord]
                        patterns_pharma_vectors[name] += [-np.array(v[idx])]
        return patterns_pharma_types, patterns_pharma_coords, patterns_pharma_vectors


    def fragment_screening(self, fragment_lib_file, alignment):
        # load fragment file
        with open(fragment_lib_file, 'r') as f:
            lines = f.readlines()
            naming = True
            points = []
            radiis = []
            atm_types = []
            fragment_lines = []
            final_lines = []
            for idx, line in enumerate(lines):
                fragment_lines += [line]
                if naming:
                    name = line.strip()
                    naming = False
                if ((len(line) == 40 and len(line.split()) == 6) or (len(line) == 70 and len(line.split()) == 16)) and \
                        line.split()[3] != 'H' and line.split()[3] != '?' and line.split()[3] != 'R' and line.split()[3] != 'R1':
                    points += [(float(line.split()[0]), float(line.split()[1]), float(line.split()[2]))]
                    try:
                        radiis += [GetPeriodicTable().GetRvdw(Chem.MolFromSmiles(line.split()[3]).GetAtoms()[0].GetAtomicNum())]
                    except:
                        print(f"ATOM type {line.split()[3]} invalid.")
                        radiis += [1]
                    atm_types += [line.split()[3]]
                if line.startswith('$$$$'):
                    feature_types, feature_coords, feature_vectors = fragment_line_to_pharmacophore(fragment_lines)
                    fragment_lines = fragment_line_rearrangement(fragment_lines)
                    origin_focus = np.mean(points, axis=0)
                    if not alignment:
                        rotated_point_vectors = points
                        best_score = get_score(points, list(self.pattern_coordinates.values()), radiis, self.pattern_density,
                                               (self.pattern_pharma_types, self.pattern_pharma_coords,
                                               self.pattern_pharma_vectors),
                                               (feature_types, feature_coords, feature_vectors),
                                               (0, 0, 0), (0, 0, 0), origin_focus)
                    else:
                        # initial alignments
                        angles = [(2*i*np.pi/4, 2*j*np.pi/4, 2*k*np.pi/4) for i in range(0, 4) for j in range(0, 4) for k in range(0, 4)]
                        rotation_vectors = rotation_vectors_(angles)
                        first_hits = rotation_vectors
                        # optimization -> Sshape + Sdipole; Powell optimization
                        second_hits = []
                        second_scores = []
                        rotation_angles = rotation_angles_(first_hits)
                        for angle in rotation_angles:
                            def my_objective(params):
                                return optimize_scoring(params, points, radiis, origin_focus,
                                             [(self.pattern_pharma_types, self.pattern_pharma_coords,
                                               self.pattern_pharma_vectors),
                                              (feature_types, feature_coords, feature_vectors)],
                                             [list(self.pattern_coordinates.values()), self.pattern_density])
                            params = [self.focus[0], self.focus[1], self.focus[2], angle[0], angle[1], angle[2]]
                            second_hit = minimize(my_objective, params, method='Powell')
                            second_hits += [second_hit.x]
                            second_scores += [-second_hit.fun]
                        best_score = max(second_scores)
                        best_score_idx = second_scores.index(best_score)
                        param = second_hits[best_score_idx]
                        translation_vector = np.array(param[:3]) - origin_focus
                        rotation_vector = rotation_vectors_([param[3:]])[0]
                        point_vectors = np.array(points) + translation_vector
                        rotated_point_vectors = rotation(np.array(param[:3]), point_vectors, rotation_vector)
                    # sdf format writing
                    final_lines += [name + '\n']
                    final_lines += fragment_lines[1:4]
                    for vector_idx, point_vector in enumerate(rotated_point_vectors):
                        final_lines += ["{:10.4f}{:10.4f}{:10.4f} {:1s} {:3d}{:3d}\n".format(point_vector[0], point_vector[1], point_vector[2], atm_types[vector_idx], 0, 0)]
                    final_lines += fragment_lines[4 + len(rotated_point_vectors):-1]
                    final_lines += [f'> <similarity>\n{round(best_score, 2)}\n\n']
                    final_lines += ['$$$$\n']
                    naming = True
                    points = []
                    radiis = []
                    atm_types = []
                    fragment_lines = []
        return final_lines


    def to_sdf(self, output):
        result = self.output_file
        w = open(output, 'w')
        for line in result:
            w.write(line)
        w.close()
        return print(f'Successfully saved {os.path.basename(output)}.')


def optimize_scoring(params, points, radiis, origin_focus, pharmacophores, pattern_info):

    focus = params[:3]
    angle = params[3:]

    TWN_pharmacophore = pharmacophores[0]
    fragment_pharmacophore = pharmacophores[1]

    pattern_coordinates = pattern_info[0]
    pattern_density = pattern_info[1]

    rotation_vector = rotation_vectors_([angle])[0]
    translation_vector = focus - origin_focus
    point_vectors = np.array(points) + translation_vector
    rotated_point_vectors = rotation(focus, point_vectors, rotation_vector)

    return -get_score(rotated_point_vectors, pattern_coordinates, radiis, pattern_density, TWN_pharmacophore, fragment_pharmacophore, translation_vector, rotation_vector, focus)


def get_score(f_coords, p_coords, f_radiis, pattern_density, p_pharm, f_pharm, t_v, r_v, focus):

    # Sshape calculation
    Sshape = 0
    for pattern_idx, pattern_coords in enumerate(p_coords):
        Sshape += get_Sshape(f_coords, pattern_coords, f_radiis) * list(pattern_density.values())[pattern_idx] / sum(list(pattern_density.values()))
    # Sdipole calculation
    Sdipole = get_Sdipole(p_pharm, f_pharm, t_v, r_v, focus, pattern_density)
    return (Sshape + Sdipole) / 2


def rotation_vectors_(angles):
    rotation_vectors = []
    for angle in angles:
        rotation_vectors += [np.round(R.from_rotvec(angle).as_matrix(), 2)]
    return np.unique(rotation_vectors, axis=0)


def rotation_angles_(vectors):
    rotation_angles = []
    for vector in vectors:
        rotation_angles += [R.from_matrix(vector).as_rotvec()]
    return rotation_angles


def rotation(focus, point_vectors, rotation_vector):
    rotated_point_vectors = []
    for vector in point_vectors:
        (x, y, z) = (vector[0] - focus[0], vector[1] - focus[1], vector[2] - focus[2])
        rotated_point_vectors += [tuple(rotation_vector.dot(np.array((x, y, z))) + np.array(focus))]
    return rotated_point_vectors


def get_Sshape(f_coords, t_coords, f_radiis):
    Oab = 0
    for t_coord in t_coords:
        Oab_part = 0
        for f_idx, f_coord in enumerate(f_coords):
            if math.dist(t_coord, f_coord) <= abs(1.55 - f_radiis[f_idx]):
                Oab_part += 4/3*np.pi*(min(1.55, f_radiis[f_idx])**3)
            elif math.dist(t_coord, f_coord) <= 1.55 + f_radiis[f_idx]:
                Oab_part += overlap_volume(f_radiis[f_idx], 1.55, math.dist(t_coord, f_coord))
        if Oab_part/1.6 <= 4/3*np.pi*(1.55**3):
            Oab += Oab_part/1.6
        else:
            Oab += 4/3*np.pi*(1.55**3)
    return Oab/(len(t_coords)*4/3*np.pi*(1.55**3))


def get_Sdipole(TWN_pharmacophore, fragment_pharmacophore, translation_vector, rotation_vector, TWN_focus, pattern_density):
    pattern_pharma_types, pattern_pharma_coords, pattern_pharma_vectors = TWN_pharmacophore
    feature_types, feature_coords, feature_vectors = fragment_pharmacophore
    Sdipole = []
    for pattern, pattern_pharma_type in pattern_pharma_types.items():
        if pattern not in pattern_density.keys():
            continue
        Sdir_patterns = []
        Sfit_patterns = []
        # Sdir & Sfit calculation
        if len(pattern_pharma_type) == 0:
            continue
        for p_idx, ppi in enumerate(pattern_pharma_type):
            Sdir_lst = []
            Sfit_lst = []
            # Hydrogen bond donor feature
            if ppi == 'HBD':
                for feature, feature_vector in feature_vectors.items():
                    if feature != 'AR':
                        Sdir_partials = [0]
                        Sfit_partials = [0]
                        for f_idx, fv in enumerate(feature_vector):
                            if feature_types[feature][f_idx] == 'HBD':
                                Sdir_partials += [np.dot(pattern_pharma_vectors[pattern][p_idx], rotation_vector.dot(fv))]
                                dist_tf = math.dist(pattern_pharma_coords[pattern][p_idx], rotation_vector.dot(feature_coords[feature][f_idx] + translation_vector - TWN_focus) + TWN_focus)
                                if dist_tf <= 3.1:
                                    Sfit_partials += [overlap_volume(1.55, 1.55, dist_tf)/(4/3*np.pi*(1.55**3))]
                                else:
                                    Sfit_partials += [0]
                        S_partials = [(Sdir_partials[i] ** (1 / 3)) * (Sfit_partials[i] ** (2 / 3)) for i in range(len(Sdir_partials))]
                        if Sdir_partials[S_partials.index(max(S_partials))] >= 0 and Sfit_partials[S_partials.index(max(S_partials))] >= 0:
                            Sdir_lst += [Sdir_partials[S_partials.index(max(S_partials))]]
                            Sfit_lst += [Sfit_partials[S_partials.index(max(S_partials))]]
                        else:
                            Sdir_lst += [0]
                            Sfit_lst += [0]
            # Hydrogen bond acceptor feature
            if ppi == 'HBA':
                for feature, feature_vector in feature_vectors.items():
                    if feature != 'AR':
                        Sdir_partials = [0]
                        Sfit_partials = [0]
                        for f_idx, fv in enumerate(feature_vector):
                            if feature_types[feature][f_idx] == 'HBA':
                                Sdir_partials += [np.dot(pattern_pharma_vectors[pattern][p_idx], rotation_vector.dot(fv))]
                                dist_tf = math.dist(pattern_pharma_coords[pattern][p_idx], rotation_vector.dot(feature_coords[feature][f_idx] + translation_vector - TWN_focus) + TWN_focus)
                                if dist_tf <= 3.1:
                                    Sfit_partials += [overlap_volume(1.55, 1.55, dist_tf)/(4/3*np.pi*(1.55**3))]
                                else:
                                    Sfit_partials += [0]
                        # print(Sdir_partials, Sfit_partials)
                        S_partials = [(Sdir_partials[i] ** (1 / 3)) * (Sfit_partials[i] ** (2 / 3)) for i in range(len(Sdir_partials))]
                        if Sdir_partials[S_partials.index(max(S_partials))] >= 0 and Sfit_partials[S_partials.index(max(S_partials))] >= 0:
                            Sdir_lst += [Sdir_partials[S_partials.index(max(S_partials))]]
                            Sfit_lst += [Sfit_partials[S_partials.index(max(S_partials))]]
                        else:
                            Sdir_lst += [0]
                            Sfit_lst += [0]
            # print(Sdir_lst, Sfit_lst)
            Stotal = [(Sdir_lst[i] ** (1 / 3)) * (Sfit_lst[i] ** (2 / 3)) for i in range(len(Sdir_lst))]
            # print(Stotal)
            Sdir_patterns += [Sdir_lst[Stotal.index(max(Stotal))] if len(Stotal) != 0 else 0]
            Sfit_patterns += [Sfit_lst[Stotal.index(max(Stotal))] if len(Stotal) != 0 else 0]
        Sdipole_patterns = []
        for Sidx, Sfit in enumerate(Sfit_patterns):
            if Sfit <= 1:
                Sdipole_patterns += [(Sfit ** (2 / 3)) * (Sdir_patterns[Sidx] ** (1 / 3))]
            else:
                Sdipole_patterns += [Sdir_patterns[Sidx] ** (1 / 3)]
        Sdipole += [np.mean(Sdipole_patterns) * pattern_density[pattern] / sum(list(pattern_density.values())) if len(Sdipole_patterns) != 0 else 0]
    # print(Sdipole)
    return sum(Sdipole)


def overlap_volume(ra, rb, dab):
    return np.pi/12*((ra + rb - dab)**2)*(dab + 2*(ra+rb) - 3/dab*((ra-rb)**2))


def vector_normalization(vector):
    return vector/np.linalg.norm(vector)


def fragment_line_rearrangement(fragment_lines):
    header_lines = fragment_lines[:3]
    atom_count, bond_count = (int(x) for x in [fragment_lines[3][0:3], fragment_lines[3][3:6]])
    atom_lines = fragment_lines[4:4+atom_count]
    bond_lines = fragment_lines[4+atom_count:4+atom_count+bond_count]
    property_lines = fragment_lines[fragment_lines.index("M  END\n")+1:]
    num_change_index = {}
    heavy_atom_lines = []
    new_bond_lines = []
    fixed_property_lines = []
    for idx, atom_line in enumerate(atom_lines):
        if atom_line.split()[3] not in ['H', 'R', 'R1', '?']:
            heavy_atom_lines += [atom_line]
            num_change_index[idx+1] = len(heavy_atom_lines)
    for bond_line in bond_lines:
        if int(bond_line.split()[0]) in num_change_index.keys() and int(bond_line.split()[1]) in num_change_index.keys():
            new_bond_lines += ["{:>3d}{:>3d}{:>3d}{:>3d}\n".format(num_change_index[int(bond_line.split()[0])], num_change_index[int(bond_line.split()[1])], int(bond_line.split()[2]), int(bond_line.split()[3]))]
    for property_line in property_lines:
        if property_line.startswith('>  <'):
            fixed_property_lines += ["> " + property_line[3:]]
        else:
            fixed_property_lines += [property_line]
    return header_lines + ["{:>3d}{:>3d}  0  0  0  0  0  0  0  0999 V2000\n".format(len(heavy_atom_lines), len(new_bond_lines))] + heavy_atom_lines + new_bond_lines + ["M  END\n"] + fixed_property_lines


def fragment_line_to_pharmacophore(fragment_lines):
    atom_count, bond_count = (int(x) for x in [fragment_lines[3][0:3], fragment_lines[3][3:6]])
    atom_lines = fragment_lines[4:4 + atom_count]
    bond_lines = fragment_lines[4 + atom_count:4 + atom_count + bond_count]
    atom_types = []
    atom_coords = []
    nodes = []
    edges = []
    # node edge generation
    for idx, atom_line in enumerate(atom_lines):
        atom_types += [atom_line.split()[3]]
        atom_coords += [(float(atom_line.split()[0]), float(atom_line.split()[1]), float(atom_line.split()[2]))]
        nodes += [idx+1]
    for bond_line in bond_lines:
        edges += [(int(bond_line.split()[0]), int(bond_line.split()[1]), int(bond_line.split()[2]))]
    feature_types = {}
    feature_coords = {}
    feature_vectors = {}
    # Hydrogen bond feature generation
    for idx, atom_line in enumerate(atom_lines):
        if atom_line.split()[3] in ['O', 'N']:
            feature_types[idx+1] = []
            feature_coords[idx+1] = []
            feature_vectors[idx+1] = []
            bond_count = 0
            bond_atoms = []
            # donors; OH, NH
            for edge in edges:
                if edge[0] == idx+1:
                    bond_count += edge[2]
                    bond_atoms += [edge[1]]
                    if atom_types[edge[1]-1] == 'H':
                        feature_types[idx+1] += ['HBD']
                        feature_coords[idx+1] += [atom_coords[idx]]
                        feature_vectors[idx+1] += [vector_normalization(np.array(atom_coords[edge[1]-1]) - np.array(atom_coords[idx]))]
                if edge[1] == idx+1:
                    bond_count += edge[2]
                    bond_atoms += [edge[0]]
                    if atom_types[edge[0]-1] == 'H':
                        feature_types[idx+1] += ['HBD']
                        feature_coords[idx+1] += [atom_coords[idx]]
                        feature_vectors[idx+1] += [vector_normalization(np.array(atom_coords[edge[0]-1]) - np.array(atom_coords[idx]))]
            # acceptors
            if atom_line.split()[3] == 'O' and bond_count == 2:
                # =O
                if len(bond_atoms) == 1:
                    bond_bond_atoms = [x[1] for x in edges if bond_atoms[0] == x[0] and idx + 1 != x[1]] + [x[0] for x in edges if bond_atoms[0] == x[1] and idx + 1 != x[0]]
                    for bond_bond_atom in bond_bond_atoms:
                        feature_types[idx+1] += ['HBA']
                        feature_coords[idx+1] += [atom_coords[idx]]
                        feature_vectors[idx+1] += [vector_normalization(np.array(atom_coords[bond_bond_atom - 1]) - np.array(atom_coords[idx]))]
                # -OH, -O-
                if len(bond_atoms) == 2:
                    vectors = []
                    for bond_atom in bond_atoms:
                        feature_types[idx + 1] += ['HBA']
                        feature_coords[idx + 1] += [atom_coords[idx]]
                        vectors += [vector_normalization(np.array(atom_coords[bond_atom - 1]) - np.array(atom_coords[idx]))]
                    # V1 + V2를 회전축으로 하여 90도 회전
                    axis = (vectors[0] + vectors[1])/np.linalg.norm((vectors[0] + vectors[1]))
                    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                    I = np.eye(3)
                    R = I + np.sin(np.pi/2) * K + (1 - np.cos(np.pi/2)) * np.dot(K, K)
                    feature_vectors[idx + 1] += [np.dot(R, x) for x in vectors]
            if atom_line.split()[3] == 'N' and (bond_count == 3 or bond_count == 8):
                # NH 제거
                if any([atom_types[x-1] == 'H' for x in bond_atoms]):
                    continue
                # -=N
                if len(bond_atoms) == 1:
                    feature_types[idx + 1] += ['HBA']
                    feature_coords[idx + 1] += [atom_coords[idx]]
                    feature_vectors[idx + 1] += [vector_normalization(np.array(atom_coords[bond_atoms[0]-1]) - np.array(atom_coords[idx]))]
                # =N-
                if len(bond_atoms) == 2:
                    feature_types[idx + 1] += ['HBA']
                    feature_coords[idx + 1] += [atom_coords[idx]]
                    feature_vectors[idx + 1] += [vector_normalization((np.array(atom_coords[bond_atoms[0]-1]) + np.array(atom_coords[bond_atoms[1]-1]))/2 - np.array(atom_coords[idx]))]
    return feature_types, feature_coords, feature_vectors


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")


def generate_output_dir(fragment_dir, base_suffix="_TWN-FS-RESULT"):
    base_dir = fragment_dir.rstrip("/")

    idx = 1
    while True:
        output_dir = f"{base_dir}{base_suffix}{idx:04d}"
        if not os.path.exists(output_dir):
            return output_dir
        idx += 1


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "TWN-FS-v2: A fragment screening workflow based on Topological Water Network (TWN) similarity.\n\n"
            "This software screens ligand fragments against predefined protein regions\n"
            "by comparing topological hydration patterns and water dipole organization."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-f", "--fragment",
        required=True,
        help="Path to the fragment directory (input fragments to be screened)"
    )

    parser.add_argument(
        "-r", "--region",
        required=True,
        help="Region definition file used for TWN analysis"
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output directory name (optional).\n"
            "If not specified, a directory named:\n"
            "  <fragment_dir>_TWN-FS-RESULT0001\n"
            "will be created automatically.\n"
            "If the directory already exists, the index will be incremented."
        )
    )

    parser.add_argument(
        "-a", "--alignment",
        type=str2bool,
        default=True,
        metavar="True/False",
        help=(
            "Enable TWN-based alignment step.\n"
            "Default: True"
        )
    )

    parser.add_argument(
        "-c", "--core",
        type=int,
        default=os.cpu_count(),
        help=(
            "Number of CPU cores to use for parallel processing.\n"
            f"Default: maximum available cores ({os.cpu_count()})"
        )
    )

    return parser


def setup_logger(output_dir, log_name="TWN-FS.log"):
    log_path = os.path.join(output_dir, log_name)

    logger = logging.getLogger("TWN")
    logger.setLevel(logging.INFO)

    # 중복 핸들러 방지 (여러 번 호출될 수 있으니 중요)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 파일 핸들러
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러 (원하면 제거 가능)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_report(logger, output_dir, similarity_key="similarity"):

    sdf_files = sorted(
        f for f in os.listdir(output_dir) if f.lower().endswith(".sdf")
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("TWN-FS-v2 Screening Result Summary")
    logger.info(f"Total output SDF files : {len(sdf_files)}")
    logger.info("=" * 60)
    logger.info("File_name\tMolecule_ID\tSimilarity")
    logger.info("-" * 60)

    total_mol = 0

    for sdf_file in sdf_files:
        sdf_path = os.path.join(output_dir, sdf_file)

        with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        mol_name = None
        similarity = "NA"
        in_prop = False

        for i, line in enumerate(lines):
            line = line.strip()

            # molecule name: first line of mol block
            if mol_name is None and line:
                mol_name = line

            # property header
            if line == f"> <{similarity_key}>":
                in_prop = True
                continue

            # property value
            if in_prop:
                similarity = line
                in_prop = False

            # end of molecule
            if line == "$$$$":
                total_mol += 1
                logger.info(f"{sdf_file}\t{mol_name}\t{similarity}")

                # reset for next molecule
                mol_name = None
                similarity = "NA"
                in_prop = False

    logger.info("-" * 60)
    logger.info(f"Total molecules reported : {total_mol}")
    logger.info("End of TWN-FS-v2 report")
    logger.info("=" * 60)
    logger.info("")


if __name__ == "__main__":
    # parsing
    parser = get_parser()
    args = parser.parse_args()
    # setting output file path
    if args.output is None:
        args.output = generate_output_dir(args.fragment)
    os.makedirs(args.output, exist_ok=False)
    # setting logs
    logger = setup_logger(args.output)
    logo = r"""
    ======================================================================
                      *-----*  ---Fragment---  //  \
                       \ TWN \  --Screening--  |   ||
                        *-----*  -using TWNs-  \\  /
                        
                      Copyright (c) Chemo-
                                    Informatics &
                                    Modeling Laboratory
                                    CNU Korea 
                                    All rights reserved.
    ======================================================================
    """
    logger.info("TWN-FS-v2 configuration")
    logger.info(logo)
    logger.info(f"Fragment directory : {args.fragment}")
    logger.info(f"Region file        : {args.region}")
    logger.info(f"Output directory   : {args.output}")
    logger.info(f"Alignment enabled  : {args.alignment}")
    logger.info(f"CPU cores used     : {args.core}")

    freeze_support()  # essential for windows multiprocessing

    # input generation
    fragment_files = [os.path.join(args.fragment, x) for x in os.listdir(args.fragment)]
    region_file = args.region
    inputs = [(region_file, x, args.output, args.alignment) for x in fragment_files]
    # run TWN-FS
    n_core = max(1, min(args.core, os.cpu_count()))
    with Pool(processes=n_core) as pool:
        results = list(tqdm(pool.imap(multi_screening, inputs), total=len(inputs)))
    # log reports
    log_report(logger, args.output)