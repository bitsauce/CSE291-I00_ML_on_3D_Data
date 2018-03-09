import argparse
import pymesh
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
parser.add_argument("--load_binary_pcl", type=bool, default=False, help="Set to true to load .pcd files [default: False]")
parser.add_argument("--generate_obj_files", type=bool, default=False, help="Set to true to generate separate .obj files for each label [default: False]")
parser.add_argument("--generate_label_files", type=bool, default=False, help="Set to true to generate a .txt file, containing the label for each face [default: False]")
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
    pcl_files[pcl_name] = pcl_file

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
    mesh = pymesh.load_mesh(mesh_file)
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
    _, indices = knn.kneighbors(mesh.vertices)

    # Create per-face labels
    per_face_label = np.zeros(len(mesh.faces), dtype=np.int32)
    for i in range(len(mesh.faces)):
        votes = [0, 0]
        # For every vertex in this triangle, count the labels
        # of the k closest points in the point cloud
        for label in labels[indices[mesh.faces[i]].flatten()]:
            votes[label] += 1
        per_face_label[i] = np.argmax(votes)
    
    if args.generate_obj_files:
        # Generate one mesh for each label
        vertices = [[] for _ in range(NUM_LABELS)]
        faces    = [[] for _ in range(NUM_LABELS)]
        for i in range(len(mesh.faces)):
            label = per_face_label[i]
            offset = len(vertices[label])
            vertices[label].extend(mesh.vertices[mesh.faces[i]])
            faces[label].append([offset, offset+1, offset+2])

        # Save meshes
        output_dir = args.output_dir + mesh_file[mesh_file.find(os.sep):mesh_file.rfind(os.sep)]
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        
        pymesh.save_mesh_raw(os.path.join(output_dir, file_name + "_s.obj"), np.array(vertices[0]), np.array(faces[0]), ascii=True)
        print("Object file created ", os.path.join(output_dir, file_name + "_l.obj"), sep="")
              
        pymesh.save_mesh_raw(os.path.join(output_dir, file_name + "_l.obj"), np.array(vertices[1]), np.array(faces[1]), ascii=True)
        print("Object file created ", os.path.join(output_dir, file_name + "_s.obj"), sep="")
        
    if args.generate_label_files:
        # Save per-face labels to file
        output_dir = args.output_dir + mesh_file[mesh_file.find(os.sep):mesh_file.rfind(os.sep)]
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        with open(os.path.join(output_dir, file_name + "_labels.txt"), "w") as f:
            f.write("Plant_labelA\n")
            for i in range(len(mesh.faces)):
                if per_face_label[i] == STEM:
                    f.write(str(i) + " ")
            f.write("\n\nPlant_labelB\n")
            for i in range(len(mesh.faces)):
                if per_face_label[i] == LEAF:
                    f.write(str(i) + " ")
            f.write("\n")
        print("Label file created ", os.path.join(output_dir, file_name + "_labels.txt"), sep="")

# Generate meshes and label files    
process_directory(args.meshes_dir, generate_labeled_meshes, ".obj")
