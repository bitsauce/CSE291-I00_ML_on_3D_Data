import os
from shutil import copyfile

def process_directory(directory, function, file_ext=None):
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            process_directory(path, function, file_ext)
        elif file_ext == None or path.lower().endswith(file_ext.lower()):
            function(path)

def copy_file(file_path):
    file_path = file_path.replace("\\", "/")
    file_name = file_path[file_path.rfind("/")+1:]
    copyfile(file_path, "Flat/%s" % file_name)

process_directory("SegmentedPlants", copy_file)
