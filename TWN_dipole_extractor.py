import math
from math import dist
import os
import logging
import argparse
from tqdm import tqdm
import subprocess
from pathlib import Path
import numpy as np


# trans format
def trans_format(form, *vals):
    # sdf format
    sdf_header = "{:s}\n  TWNPATTERN          3D                             0\n"
    sdf_count = "{:3d}{:3d}  0  0  0  0  0  0  0  0999 V2000"
    sdf_atom = "{:10.4f}{:10.4f}{:10.4f} {:1s} {:3d}{:3d}"
    sdf_bond = "{:3d}{:3d}{:3d}  0  0  0"
    sdf_prop_s = "> <{:s}>\n{:s}\n"
    sdf_prop_d = "> <{:s}>\n{:d}\n"
    #pdb_format
    pdb_line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}"
    pdb_ter = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}"
    pdb_model = '{:6s}     {:>3d}'
    trans = {'sdf_header': sdf_header, 'sdf_count': sdf_count, 'sdf_atom': sdf_atom,
             'sdf_bond': sdf_bond, 'sdf_prop_s': sdf_prop_s, 'sdf_prop_d': sdf_prop_d,
             'pdb_line': pdb_line, 'pdb_ter': pdb_ter, 'pdb_model': pdb_model}
    return trans[form].format(*vals)


def boundary_reader(boundary_file):
    boundary_inform = []
    with open(boundary_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('METHOD'):
                boundary_inform += ["_".join(line.split()[1:]).rstrip()]
            if line.startswith('RANGE'):
                boundary_inform += [float(line.split()[1].strip())]
            if line.startswith('COORD'):
                boundary_inform += [[int(line.split()[1]), tuple((float(line.split()[2].strip()), float(line.split()[3].strip()), float(line.split()[4].strip())))]]
            if line.startswith('RESIDUE'):
                boundary_inform += ["-".join(line.split()[1:]).rstrip()]
    return boundary_inform


def pdb_spliter(pdb_line):
    type = pdb_line[0:6].strip()
    atm_index = pdb_line[6:11].strip()
    atm_type = pdb_line[11:17].strip()
    residue_name = pdb_line[17:20].strip()
    chain = pdb_line[21:22].strip()
    residue_index = pdb_line[22:26].strip()
    x_coord = pdb_line[30:38].strip()
    y_coord = pdb_line[38:46].strip()
    z_coord = pdb_line[46:54].strip()
    frequency = pdb_line[54:60].strip()
    b_factor = pdb_line[60:66].strip()
    element = pdb_line[66:].strip()
    return [type, atm_index, atm_type, residue_name, chain, residue_index,
            x_coord, y_coord, z_coord, frequency, b_factor, element]


def sdf_reader(sdf_file):
    point_dicts = {}
    with open(sdf_file, 'r') as f:
        lines = f.readlines()
        naming = True
        points = []
        for line in lines:
            if naming:
                name = line.strip()
                naming = False
            if len(line) == 40 and len(line.split()) == 6 and line.split()[3] != 'H' and line.split()[3] != '?':
                points += [np.array((float(line.split()[0]), float(line.split()[1]), float(line.split()[2])))]
            if line.startswith('$$$$'):
                point_dicts[name] = points
                points = []
                naming = True
    return point_dicts


# read protein
def trajectory_reader(trajectory_file, boundary_file):
    # boundary setting
    boundary_inform = boundary_reader(boundary_file)
    centers = []
    residue_centers = []
    if boundary_inform[0] != 'residue_center_extraction':
        for trj, coord in boundary_inform[2:]:
            if trajectory_file.stem == str(trj):
                centers += [coord]
    # get protein inform
    with open(trajectory_file, 'r') as f:
        lines = f.readlines()
        protein_inform = []
        for line in lines:
            line = line.rstrip()
            atoms = pdb_spliter(line)

            if atoms[0] != 'ATOM':
                protein_inform += [line]
                continue

            elif atoms[3] == 'SOL' or atoms[3] == 'CL' or atoms[3] == 'NA':
                continue

            else:
                if boundary_inform[0] == 'residue_center_extraction' and (atoms[3] + atoms[5]) in boundary_inform[2].split("-") and atoms[2] == 'CA':
                    residue_centers += [tuple((float(atoms[6]), float(atoms[7]), float(atoms[8])))]
                pdb_row = trans_format('pdb_line', atoms[0], int(atoms[1]), atoms[2], '', atoms[3], atoms[4], int(atoms[5]), '',
                                       float(atoms[6]), float(atoms[7]), float(atoms[8]),
                                       float(atoms[9]), float(atoms[10]), atoms[11], '')
                protein_inform += [pdb_row]
        if len(residue_centers) != 0:
            centers = [np.mean(residue_centers, axis=0)]
        water_inform = []
        first_atm = True
        ht = 0
        for idx, line in enumerate(lines):
            line = line.rstrip()
            atoms = pdb_spliter(line)

            if atoms[3] == 'SOL':
                # save atom number
                if int(atoms[1]) == 0:
                    ht += 100000
                atm_num = int(atoms[1]) + ht
                # define first atom
                if first_atm:
                    first_num = atm_num
                    first_atm = False
                if atoms[2] == 'OW':
                    w_num = int((atm_num-first_num)/3 + 1)
                    if any([dist((float(atoms[6]), float(atoms[7]), float(atoms[8])), center) <= boundary_inform[1] for center in centers]):
                        water_row = trans_format('pdb_line', atoms[0], int(atoms[1]), atoms[2], '', 'W' + "{:0>6d}".format(w_num)[:2], '', int("{:0>6d}".format(w_num)[2:]), '',
                                                 float(atoms[6]), float(atoms[7]), float(atoms[8]),
                                                 float(atoms[9]), float(atoms[10]), atoms[11], '')
                        h1_atoms = pdb_spliter(lines[idx+1])
                        h2_atoms = pdb_spliter(lines[idx+2])
                        h1_row = trans_format('pdb_line', h1_atoms[0], int(h1_atoms[1]), h1_atoms[2], '', 'W' + "{:0>6d}".format(w_num)[:2], '', int("{:0>6d}".format(w_num)[2:]), '',
                                                 float(h1_atoms[6]), float(h1_atoms[7]), float(h1_atoms[8]),
                                                 float(h1_atoms[9]), float(h1_atoms[10]), h1_atoms[11], '')
                        h2_row = trans_format('pdb_line', h2_atoms[0], int(h2_atoms[1]), h2_atoms[2], '',
                                              'W' + "{:0>6d}".format(w_num)[:2], '', int("{:0>6d}".format(w_num)[2:]),
                                              '',
                                              float(h2_atoms[6]), float(h2_atoms[7]), float(h2_atoms[8]),
                                              float(h2_atoms[9]), float(h2_atoms[10]), h2_atoms[11], '')
                        water_inform += [water_row] + [h1_row] + [h2_row]

    return protein_inform, water_inform


def get_pattern_pharmacophore(pattern_file, trajectory_files, boundary_file):
    # Load trajectory information including dipole
    print('Loading trajectories...')
    single_box = {}
    for trajectory_file in tqdm(trajectory_files):
        trajectory = trajectory_file.stem
        file_box = []
        lines = trajectory_reader(trajectory_file, boundary_file)[1]
        for idx, line in enumerate(lines):
            atoms = pdb_spliter(line)
            if atoms[2] != 'OW':
                continue
            if len(atoms) == 11:
                atoms += ['O']
            if len(atoms) != 12:
                continue
            o_coord = float(atoms[6]), float(atoms[7]), float(atoms[8])
            h1_coord = float(lines[idx + 1][30:38].strip()), float(lines[idx + 1][38:46].strip()), float(lines[idx + 1][46:54].strip())
            h2_coord = float(lines[idx + 2][30:38].strip()), float(lines[idx + 2][38:46].strip()), float(lines[idx + 2][46:54].strip())
            h_center = np.mean([h1_coord, h2_coord], axis=0)
            dipole = np.array((o_coord - h_center) / np.linalg.norm(o_coord - h_center))
            single_data = [atoms[1], atoms[2], trajectory, atoms[3] + atoms[5],
                           float(atoms[6]), float(atoms[7]), float(atoms[8]),
                           float(atoms[9]), float(atoms[10]), atoms[11], dipole[0], dipole[1], dipole[2]]
            file_box += [single_data]
        single_box[trajectory] = file_box
    print('Start to generate pharmacophores...')
    # Pharmacophore generation
    pharmacophore_lines = ['REMARK     TWN-pharma file\n\n']
    point_dicts = sdf_reader(pattern_file)
    for pattern, coords in tqdm(point_dicts.items()):
        pattern_lines = [f'{pattern}\n']
        feature_lines = []
        HB_count = 0
        Ar_count = 0
        # SDO -> HB
        for coord in coords:
            SDO_magnitude, SDO = SDO_calculation(coord, single_box)
            SDO_dv = SDO/SDO_magnitude
            if SDO_magnitude >= 0.7:
                HB_count += 1
                feature_lines += ["{:6s}{:5d} {:8.3f}{:8.3f}{:8.3f}  {:2s} {:6d}   {:8.3f}{:8.3f}{:8.3f}\n".format("FEAT", HB_count+Ar_count, coord[0], coord[1], coord[2], "HB", int(pattern.split("_")[-1]), SDO_dv[0], SDO_dv[1], SDO_dv[2])]
        # Normal Vector -> AR
        aromatic_point, aromatic_dv = aromatic_calculation(coords)
        Ar_count += 1
        feature_lines += ["{:6s}{:5d} {:8.3f}{:8.3f}{:8.3f}  {:2s} {:6d}   {:8.3f}{:8.3f}{:8.3f}\n".format("FEAT", HB_count+Ar_count, aromatic_point[0], aromatic_point[1], aromatic_point[2], "AR", int(pattern.split("_")[-1]), aromatic_dv[0], aromatic_dv[1], aromatic_dv[2])]
        pattern_lines += ["TOT{:4d} HB{:4d} AR{:4d}\n".format(HB_count+Ar_count, HB_count, Ar_count)]
        pattern_lines += feature_lines
        pattern_lines += ["$$$$\n\n"]
        pharmacophore_lines += pattern_lines
    return pharmacophore_lines


def aromatic_calculation(points):
    distances = {}
    for i1, p1 in enumerate(points):
        for i2, p2 in enumerate(points):
            if i2 > i1:
                distances[f'{i1}-{i2}'] = math.dist(p1, p2)
    sorted_distances = dict(sorted(distances.items(), key=lambda x: x[1], reverse=True))
    x1, y1 = (int(x) for x in list(sorted_distances.keys())[0].split('-'))
    x2, y2 = (int(x) for x in list(sorted_distances.keys())[1].split('-'))
    aromatic_point = np.mean(points, axis=0)
    normal_vector = np.cross(points[x1] - points[y1], points[x2] - points[y2]) / np.linalg.norm(np.cross(points[x1] - points[y1], points[x2] - points[y2]))
    return aromatic_point, normal_vector


def SDO_calculation(coord, single_box):
    dipoles = []
    for trj, atoms in single_box.items():
        for atom in atoms:
            if math.dist(coord, (atom[4], atom[5], atom[6])) <= 1:
                dipoles += [(atom[10], atom[11], atom[12])]
    SDO = np.mean(dipoles, axis=0)
    SDO_magnitude = np.linalg.norm(SDO)
    return SDO_magnitude, SDO




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dipole calculation')
    parser.add_argument('-d', '--directory', required=True, help='Set your TWN-Pattern directory to analyze')
    args = parser.parse_args()

    # Pattern identification
    path = f"{args.directory}".replace("\\", "/")
    proteins = os.listdir(path)
    trajectory_path = "/".join(path.split("/")[:-1]) + "/trajectory"
    boundary_path = "/".join(path.split("/")[:-1]) + "/boundary"
    output_path = "/".join(path.split("/")[:-1]) + "/TWN-pharma"
    os.mkdir(output_path)
    for protein in proteins:
        os.mkdir(Path(output_path + '/' + protein))
        pattern_file = Path(path + '/' + protein + '/TWN.sdf')
        trajectory_files = [trj for trj in Path(trajectory_path + '/' + protein.split("_")[1] + '/a_input').glob('./*.pdb')]
        boundary_file = Path(boundary_path + '/' + protein.split("_")[1] + '/Center.bd')
        lines = get_pattern_pharmacophore(pattern_file, trajectory_files, boundary_file)
        w = open(output_path + '/' + protein + '/Pharmacophore.phm', 'w')
        for line in lines:
            w.write(line)
        w.close()
        print(f'Finish to generate pharmacophore for {protein}')