import argparse
#import pcl
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import sys

# Add the path of this file to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--obj_file", help="", required=True)
parser.add_argument("--labels_file", help="", required=True)
parser.add_argument("--export_group_obj", type=bool, help="", required=True)
args = parser.parse_args()

#############################################################
# def process_directory:
# Recursively iterate all files in directory and call
# 'function' on all files ending in 'file_ext'
#############################################################
def process_directory(directory, function, file_ext=None):
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            process_directory(path, function, file_ext)
        elif file_ext == None or path.lower().endswith(file_ext.lower()):
            function(path)

def save_obj(out_file, vertices, normals, faces, group_names=None):
    with open(out_file, "w") as f:
        if group_names != None:
            for i in range(len(vertices)):
                f.write("g %s\n" % group_names[i])
                for v in vertices[i]: f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
                for vn in normals[i]: f.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))
                for v1, v2, v3 in faces[i]: f.write("f {}//{} {}//{} {}//{}\n".format(v1+1, v1+1, v2+1, v2+1, v3+1, v3+1))
        else:
            for v in vertices: f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            for vn in normals: f.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))
            for v1, v2, v3 in faces: f.write("f {}//{} {}//{} {}//{}\n".format(v1+1, v1+1, v2+1, v2+1, v3+1, v3+1))

def load_obj(in_file):
    vertices = []
    normals  = []
    faces    = []
    with open(in_file, "r") as f:
        for line in f.readlines():
            if line.startswith("v "): vertices.append([float(v) for v in line[2:].strip().split()])
            elif line.startswith("vn "): normals.append([float(vn) for vn in line[3:].strip().split()])
            elif line.startswith("f "): faces.append([int(f[:f.find("/")])-1 for f in line[2:].strip().split()])
    vertices = np.array(vertices)
    normals = np.array(normals)
    faces = np.array(faces)
    return vertices, normals, faces

# Pre-defined labels
STEM = 0
LEAF = 1
NUM_LABELS = 2

def generate_labeled_meshes(mesh_file, labels_file):    
    print("Processing ", mesh_file, "...", sep="")
    
    # Load mesh and its labeled point cloud
    mesh_vertices, mesh_normals, mesh_faces = load_obj(mesh_file)

    # Create per-face labels
    per_face_label = pd.read_csv(labels_file).as_matrix()

    if False:#args.export_group_obj:
        # Generate one mesh with 2 groups
        vertices = [[] for _ in range(NUM_LABELS)]
        normals  = [[] for _ in range(NUM_LABELS)]
        faces    = [[] for _ in range(NUM_LABELS)]
        index = 0
        vertex_map = {}
        for label in [STEM, LEAF]:
            for i in range(len(per_face_label)):
                if per_face_label[i][0] == label:
                    face = []
                    for vertex, normal in zip(mesh_vertices[mesh_faces[i]], mesh_normals[mesh_faces[i]]):
                        vertex_key = tuple(vertex)
                        if vertex_key in vertex_map:
                            face.append(vertex_map[vertex_key])
                        else:
                            vertices[label].append(vertex)
                            normals[label].append(normal)

                            vertex_map[vertex_key] = index
                            face.append(index)
                            index += 1
                    faces[label].append(face)

        # To np.array
        vertices = [np.array(vertices[i]) for i in range(NUM_LABELS)]
        normals = [np.array(normals[i]) for i in range(NUM_LABELS)]
        faces = [np.array(faces[i]) for i in range(NUM_LABELS)]

        # Save meshes
        #output_dir = args.output_dir + mesh_file[mesh_file.find(os.sep):mesh_file.rfind(os.sep)]
        #if not os.path.isdir(output_dir): os.makedirs(output_dir)
        
        #save_obj(os.path.join(output_dir, file_name + ".obj"), vertices, normals, faces, group_names=["Plant_stem", "Plant_leaf"])
        #print("Object file created ", os.path.join(output_dir, file_name + ".obj"), sep="")\
    else:
        # Generate one mesh for each label
        for label in [STEM, LEAF]:
            vertices = []
            normals  = []
            faces    = []
            vertex_map = {}
            for i in range(len(per_face_label)):
                if per_face_label[i][0] == label:
                    face = []
                    for vertex, normal in zip(mesh_vertices[mesh_faces[i]], mesh_normals[mesh_faces[i]]):
                        vertex_key = tuple(vertex)
                        if vertex_key in vertex_map:
                            face.append(vertex_map[vertex_key])
                        else:
                            vertices.append(vertex)
                            normals.append(normal)

                            index = len(vertices) - 1
                            vertex_map[vertex_key] = index
                            face.append(index)
                    faces.append(face)

            # To np.array
            vertices = np.array(vertices)
            normals = np.array(normals)
            faces = np.array(faces)

            # Save meshes
            #output_dir = args.output_dir + mesh_file[mesh_file.find(os.sep):mesh_file.rfind(os.sep)]
            #if not os.path.isdir(output_dir): os.makedirs(output_dir)
            
            #save_obj(os.path.join(output_dir, file_name + "_%s.obj" % ("s" if label == STEM else "l")), vertices, normals, faces)
            #print("Object file created ", os.path.join(output_dir, file_name + "_%s.obj" % ("s" if label == STEM else "l")), sep="")
            save_obj("seg_%s.obj" % ("s" if label == STEM else "l"), vertices, normals, faces)
            print("Saved")

# Generate meshes and label files    
generate_labeled_meshes(args.obj_file, args.labels_file)
