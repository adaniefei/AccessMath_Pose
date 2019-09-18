import csv
import pandas as pd

# read from csv file from column names or c
'''
def readfile(file_path, colname):
    if colname == []:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, usecols=colname)
    return df
'''

# create a csv file given path and columns name
def write2file(file_path, csv_col_name):
    with open(file_path, 'w', newline='') as outfile:
        outfile_writer = csv.writer(outfile)
        outfile_writer.writerow(csv_col_name)


# add row of data into csv file
def add2csv(file_path, row_data):
    with open(file_path, 'a', newline='') as outfile:
        outfile_writer = csv.writer(outfile)
        outfile_writer.writerow(row_data)


# combine json files into one csv file
def jsons2csv(json_list, csv_path, csv_col_name):
    write2file(csv_path, csv_col_name)
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
            add2csv(csv_path, row_data)
    return