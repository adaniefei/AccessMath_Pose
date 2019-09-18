# basic ops for general files
import os


class PlayFiles:
    class SearchCond:
        def __init__(self):
            self.head = ""
            self.tail = ""
            self.interven = ""

    def __init__(self, path):
        self.path = path

    # Safely create a directory
    def create_dir(self, path=""):
        if len(path.strip()) == 0:
            path = self.path
        if not os.path.isdir(path):
            print("Create directory: {0}".format(path))
            os.mkdir(path)
        return path

    # find the sub-directories of a given path
    def find_subdir(self, path=""):
        if len(path.strip()) == 0:
            path = self.path
        return list(filter(os.path.isdir, [os.path.join(path, f) for f in sorted(os.listdir(path))]))

    # Get list of files given one path and searching conditions
    def search_file_in_path(self, cond, path=""):
        if len(path.strip()) == 0:
            path = self.path
        file_list = []

        # save all videos absolute path in "video_list"
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith(cond.head) and file.lower().endswith(cond.tail) and cond.interven in file:
                    file_list.append(os.path.join(root, file))
        return sorted(file_list)

