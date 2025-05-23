from mesh_to_sdf import sample_sdf_near_surface

import os
import glob
import trimesh
import numpy as np


def generate_xyz_sdf(filename):
    mesh = trimesh.load(os.path.abspath(filename),force='mesh')
    xyz, sdf = sample_sdf_near_surface(mesh, number_of_points=15000)
    return xyz, sdf

def writeSDFToNPZ(xyz, sdfs, filename):
    num_vert = len(xyz)
    pos = []
    neg = []

    for i in range(num_vert):
        v = xyz[i]
        s = sdfs[i]

        if s > 0:
            for j in range(3):
                pos.append(v[j])
            pos.append(s)
        else:
            for j in range(3):
                neg.append(v[j])
            neg.append(s)
    
    np.savez(filename, pos=np.array(pos).reshape(-1, 4), neg=np.array(neg).reshape(-1, 4))
    
def process(mesh_filepath, target_filepath):
    xyz, sdfs = generate_xyz_sdf(mesh_filepath)
    writeSDFToNPZ(xyz, sdfs, target_filepath)
    

class_path = "/validation/"
target_path = "./processed_data/validation/"

isExist = os.path.exists(target_path)

if not isExist:
   os.makedirs(target_path)

# find all mesh file from dataset
mesh_filenames = list(glob.iglob("dataset" + class_path + "*.obj"))

N = len(mesh_filenames)
it = 0

for mesh_filepath in mesh_filenames:
    filename = os.path.basename(mesh_filepath)
    target_filename = os.path.splitext(filename)[0]
    target_filepath = os.path.join(target_path, target_filename)

    # generate point clouds
    process(mesh_filepath, target_filepath)

    it += 1
    print("process finished:", filename, it, "/", N)