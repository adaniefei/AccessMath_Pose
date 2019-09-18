import json
from .genfile import PlayFiles


# write/create a file given path
def write2file(data, file_path):
    with open(jfile_path, 'w') as outfile:
        json.dump(self, data, outfile)

# read a file given path
def read_file(data, file_path):
    with open(jfile_path, 'r') as infile:
        data = json.load(infile)
    return data

# add data into a file
def add2file(data, file_path):
    with open(jfile_path, 'a') as outfile:
        json.dump(self, data, outfile)

# get a list of all json files from a folder
def get_json_list(dir):
    # Get the json file directory
    jf_dir = args.in_dir
    pf = PlayFiles(jf_dir)
    cond = pf.SearchCond()
    cond.tail = ".json"
    # Get the file list of all json files given directory
    jf_list = pf.search_file_in_path(cond, jf_dir)
    return jf_list


