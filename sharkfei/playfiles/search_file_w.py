import os



# Go through directory with condition to get list of files
def search_file_in_path(path, cond):
    file_head = cond.head
    file_end = cond.end
    file_interven = cond.interven

    file_list = []

    # save all videos absolute path in "video_list"
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(file_head) and file.endswith(file_end) and file_interven in file:
                file_list.append(os.path.join(root, file))
    return file_list
