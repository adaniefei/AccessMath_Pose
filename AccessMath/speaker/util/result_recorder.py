
import csv


class ResultRecorder:
    def __init__(self, file_path):
        self.file_path = file_path

    def write_headers(self, csv_col_name):
        with open(self.file_path, 'w', newline='') as outfile:
            outfile_writer = csv.writer(outfile)
            outfile_writer.writerow(csv_col_name)

    def record_results(self, values):
        with open(self.file_path, 'a', newline='') as outfile:
            outfile_writer = csv.writer(outfile)
            for ind in range(0, len(values[0])):
                row = []
                for cat in range(0, len(values)):
                    add_in = values[cat][ind]
                    if not isinstance(add_in, list):
                        add_in = [add_in]
                    row += add_in

                outfile_writer.writerow(row)
