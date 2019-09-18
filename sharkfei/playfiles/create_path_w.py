import os

# Safely create a directory give path
def create_dir(path):
    if not os.path.isdir(path):
        print("Create directory: {0}".format(path))
        os.mkdir(path)
    return

