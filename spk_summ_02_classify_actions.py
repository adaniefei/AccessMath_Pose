
import os
import sys

import numpy as np

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.speaker.util.result_recorder import ResultRecorder

def main():
    if len(sys.argv) < 2:
        print("Usage")
        print("\tpython {0:s} config".format(sys.argv[0]))
        return

    # initialization #
    config = Configuration.from_file(sys.argv[1])

    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid AccessMath Database file")
        return

    # get paths and other configuration parameters ....
    output_dir = config.get_str("OUTPUT_PATH")
    features_dir = output_dir + "/" + config.get("SPEAKER_ACTION_FEATURES_DIR")

    action_class_output_dir = output_dir + "/" + config.get("SPEAKER_ACTION_CLASSIFICATION_OUTPUT_DIR")
    action_class_probabilities_dir = output_dir + "/" + config.get("SPEAKER_ACTION_CLASSIFICATION_PROBABILITIES_DIR")
    os.makedirs(action_class_output_dir, exist_ok=True)
    os.makedirs(action_class_probabilities_dir, exist_ok=True)

    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    # load the saved model for action classification ...
    classifier_dir = output_dir + "/" + config.get_str("SPEAKER_ACTION_CLASSIFIER_DIR")
    classifier_filename = classifier_dir + "/" + config.get_str("SPEAKER_ACTION_CLASSIFIER_FILENAME")
    clf = MiscHelper.dump_load(classifier_filename)

    csv_col = ['frame_start', 'frame_end', 'prediction']

    for lecture in testing_set:
        input_filename = features_dir + "/" + database.name + "_" + lecture.title + ".pickle"
        output_actions_filename = action_class_output_dir + "/" + database.name + "_" + lecture.title + ".csv"
        output_proba_filename = action_class_probabilities_dir + "/" + database.name + "_" + lecture.title + ".csv"

        # load data ...
        data_xy = MiscHelper.dump_load(input_filename)

        # classier predict ....
        test_x = data_xy["features"]
        y_pred = clf.predict(test_x)
        y_pred_re = y_pred.reshape((y_pred.shape[0], 1))

        # save prediction result
        output_csv = ResultRecorder(output_actions_filename)
        output_csv.write_headers(csv_col)

        # the function accepts a list of columns to save on CSV ...
        # by transposing, we make the standard list of rows into a list of columns for the function ...
        paras = np.hstack((data_xy["frame_infos"], y_pred[:, None])).transpose()
        output_csv.record_results(paras)

        # save label probabilities
        all_classes = clf.classes_
        y_prob = clf.predict_proba(test_x)
        infos = np.concatenate((y_pred_re, y_prob), axis=1)
        output_csv = ResultRecorder(output_proba_filename)
        output_csv.write_headers(csv_col + all_classes.tolist())
        # ... IDEM ....
        paras = np.hstack((data_xy["frame_infos"], infos)).transpose()
        output_csv.record_results(paras)


if __name__ == '__main__':
    main()

