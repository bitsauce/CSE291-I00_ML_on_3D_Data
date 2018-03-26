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
parser.add_argument("--meshes_dir", help="Directory containing meshes", required=True)
parser.add_argument("--pcl_dir", help="Directory containing labeled point clouds", required=True)
parser.add_argument("--output_dir", help="Output directory", required=True)
parser.add_argument("--export_group_obj", type=bool, help="If True, export a single OBJ file with vertex grouping", required=True)
parser.add_argument("--load_binary_pcl", type=bool, default=False, help="Set to True to load .pcd files [default: False]")
parser.add_argument("--k", type=int, default=3, help="K to be used in k-nearest neightbours [default: 3]")
args = parser.parse_args()

# Pre-defined labels
STEM = 0
LEAF = 1
NUM_LABELS = 2

# Global mapping from file name to pcl file
# Populated by registerPointCloudFile()
pcl_files = dict()

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

#############################################################
# def get_file_name:
# Returns the name of the file given by a file path
# e.g. "/alpha/beta/gamma.ext" returns "gamma"
#############################################################
def get_file_name(filepath):
    return os.path.basename(filepath).split(".")[0]

#############################################################
# def registerPointCloudFile:
# Maps the name of a point cloud file to its full path
# in the global map 'pcl_files'
#############################################################
def registerPointCloudFile(pcl_file):
    global pcl_files
    pcl_name = get_file_name(pcl_file).lower()
    if pcl_name.startswith("ctrl"):
        pcl_name = "control" + pcl_name[4:]
    if pcl_name in pcl_files: raise Exception("Multiple files with the same name %s" % pcl_name)
    pcl_files[pcl_name] = pcl_file

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
    
# Find and store all pcl files
if args.load_binary_pcl:
    process_directory(args.pcl_dir, registerPointCloudFile, ".pcd")
else:
    process_directory(args.pcl_dir, registerPointCloudFile, ".txt")

#############################################################
# def generate_labeled_meshes:
# Given a mesh file, load the corresponding pcl file
# and label the faces of the mesh given the k nearest
# neigbours in the labeled point cloud
#############################################################
def generate_labeled_meshes(mesh_file):
    # Find corresponding pcl file for the mesh
    file_name = get_file_name(mesh_file)
    pcl_stem_file = file_name.lower() + "_s"
    pcl_leaf_file = file_name.lower() + "_l"
    if pcl_stem_file not in pcl_files or pcl_leaf_file not in pcl_files:
        print("No point cloud file found for mesh", mesh_file)
        return
    
    print("Processing ", mesh_file, "...", sep="")
    
    # Load mesh and its labeled point cloud
    mesh_vertices, mesh_normals, mesh_faces = load_obj(mesh_file)
    if args.load_binary_pcl:
        pcl_stem = pcl.io.PCDReader().read(pcl_files[pcl_stem_file])
        pcl_leaf = pcl.io.PCDReader().read(pcl_files[pcl_leaf_file])
    else:
        pcl_stem = pd.read_csv(pcl_files[pcl_stem_file], sep=",", names=["x", "y", "z"]).as_matrix()
        pcl_leaf = pd.read_csv(pcl_files[pcl_leaf_file], sep=",", names=["x", "y", "z"]).as_matrix()
    
    # Create an array which gives the label of every vertex
    # in the combined stem and leaf point clouds
    labels = np.concatenate((np.full((pcl_stem.shape[0]), STEM),
                             np.full((pcl_leaf.shape[0]), LEAF)))

    # Concatenate the stem and leaf point clouds
    points = np.concatenate((pcl_stem, pcl_leaf))
    
    # For every vertex, find the k nearest points in the point cloud
    knn = NearestNeighbors(n_neighbors=args.k, algorithm='ball_tree').fit(points)
    _, indices = knn.kneighbors(mesh_vertices)

    # Create per-face labels
    per_face_label = np.zeros(len(mesh_faces), dtype=np.int32)
    for i in range(len(mesh_faces)):
        votes = [0, 0]
        # For every vertex in this triangle, count the labels
        # of the k closest points in the point cloud
        for label in labels[indices[mesh_faces[i]].flatten()]:
            votes[label] += 1
        per_face_label[i] = np.argmax(votes)
    
    if not args.export_group_obj:
        # Generate one mesh for each label
        for label in [STEM, LEAF]:
            vertices = []
            normals  = []
            faces    = []
            vertex_map = {}
            for i in range(len(mesh_faces)):
                if per_face_label[i] == label:
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
            output_dir = args.output_dir + mesh_file[mesh_file.find(os.sep):mesh_file.rfind(os.sep)]
            if not os.path.isdir(output_dir): os.makedirs(output_dir)
            
            save_obj(os.path.join(output_dir, file_name + "_%s.obj" % ("s" if label == STEM else "l")), vertices, normals, faces)
            print("Object file created ", os.path.join(output_dir, file_name + "_%s.obj" % ("s" if label == STEM else "l")), sep="")
    else:
        # Generate one mesh with 2 groups
        vertices = [[] for _ in range(NUM_LABELS)]
        normals  = [[] for _ in range(NUM_LABELS)]
        faces    = [[] for _ in range(NUM_LABELS)]
        index = 0
        vertex_map = {}
        for label in [STEM, LEAF]:
            for i in range(len(mesh_faces)):
                if per_face_label[i] == label:
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
        output_dir = args.output_dir + mesh_file[mesh_file.find(os.sep):mesh_file.rfind(os.sep)]
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        
        save_obj(os.path.join(output_dir, file_name + ".obj"), vertices, normals, faces, group_names=["Plant_stem", "Plant_leaf"])
        print("Object file created ", os.path.join(output_dir, file_name + ".obj"), sep="")

# Generate meshes and label files    
process_directory(args.meshes_dir, generate_labeled_meshes, ".obj")
