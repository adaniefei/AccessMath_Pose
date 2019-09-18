
import os
import sys
import json
from sharkfei.playfiles.genfile import PlayFiles
import argparse

import csv

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess

def get_col_names(prefix, n_items):
    x_prefix = prefix + "_x"
    y_prefix = prefix + "_y"
    c_prefix = prefix + "_c"

    col_names = []
    for idx in range(n_items):
       col_names.extend([x_prefix + str(idx), y_prefix + str(idx), c_prefix + str(idx)])

    return col_names


def add2csv(csv_path, row_data):
    with open(csv_path, 'a', newline="") as outfile:
        outfile_writer = csv.writer(outfile)
        outfile_writer.writerow(row_data)
    return


def combine_json_data(json_list, output_path):
    # for cases where no skeleton data is found .... (25 body kp + 21 on each hand) times (x, y, conf)
    except_val = [-1000] * (25 + 21 + 21) * 3

    for index in range(len(json_list)):
        with open(json_list[index], 'r') as infile:
            print("Current json file {0}".format(json_list[index]))
            data = json.load(infile)
            data_sub = data['people']
            if len(data_sub) == 0:
                row_data = [index] + except_val
            else:
                data_sub = data['people'][0]
                pose_2d = data_sub["pose_keypoints_2d"]
                hand_left_2d = data_sub["hand_left_keypoints_2d"]
                hand_right_2d = data_sub["hand_right_keypoints_2d"]
                row_data = [index]
                row_data += pose_2d
                row_data += hand_left_2d
                row_data += hand_right_2d
            add2csv(output_path, row_data)
    return


def video_process(process, input_data):
    # Get the json file directory
    lecture_prefix = process.database.name + "_" + process.current_lecture.title

    output_main_dir = process.configuration.get_str("OUTPUT_PATH")
    rel_json_dir = process.configuration.get_str("OPENPOSE_OUTPUT_DIR_JSON")
    rel_csv_dir = process.configuration.get_str("OPENPOSE_OUTPUT_DIR_CSV")
    json_root_dir = output_main_dir + "/" + rel_json_dir + "/" + lecture_prefix
    jf_dir = json_root_dir + "/" + 'json'

    pf = PlayFiles(jf_dir)
    cond = pf.SearchCond()
    cond.tail = ".json"
    # Get the file list of all json files given directory
    jf_list = pf.search_file_in_path(cond, jf_dir)

    output_dir = output_main_dir + "/" + rel_csv_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + "/" + lecture_prefix + ".csv"

    csv_col_name = ["Frame"]
    csv_col_name += get_col_names("body", 25)
    csv_col_name += get_col_names("lh", 21)
    csv_col_name += get_col_names("rh", 21)

    # Create output csv file
    with open(output_path, 'w', newline="") as outfile:
        outfile_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        outfile_writer.writerow(csv_col_name)

    # Add json data into csv files
    combine_json_data(jf_list, output_path)


def main():
    # usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
        return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], None, None)
    if not process.initialize():
        return

    process.start_input_processing(video_process)


if __name__ == "__main__":
    main()
