
import os
import sys

import numpy as np

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.speaker.util.result_recorder import ResultRecorder
from AccessMath.speaker.actions.pose_feature_extractor import PoseFeatureExtractor

from sklearn.ensemble import RandomForestClassifier




def print_confusion_matrix(unique_label, conf_matrix, title_str):
    print("")
    print("Confusion matrix for: " + title_str)

    longest = max([len(label) for label in unique_label])

    total_correct = 0
    total_samples = 0
    for label in unique_label:
        row = label + (" " * (longest - len(label)))
        for pred_label in unique_label:
            row += "\t" + str(conf_matrix[label][pred_label])
            total_samples += conf_matrix[label][pred_label]
            if label == pred_label:
                total_correct += conf_matrix[label][pred_label]
        print(row)
    acc = total_correct * 100.0 / total_samples
    print("Accuracy\t{0:.2f}".format(acc))

    return acc


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

    dataset_name = config.get("SPEAKER_TRAINING_SET_NAME")
    training_set = database.datasets[dataset_name]
    training_titles = [lecture.title.lower() for lecture in training_set]

    unique_label = config.get("SPEAKER_VALID_ACTIONS")

    output_csv_filename = output_dir + "/" + config.get_str("SPEAKER_ACTION_CROSSVALIDATION_RESULT")
    csv_col = ['vid_name', 'frame_start', 'frame_end', 'ground_truth', 'prediction']
    spreadsheet = ResultRecorder(output_csv_filename)
    spreadsheet.write_headers(csv_col)

    label_probabilities_dir = output_dir + "/" + config.get("SPEAKER_ACTION_LABEL_PROBABILITIES_DIR")
    os.makedirs(label_probabilities_dir, exist_ok=True)

    # read all training data available ....
    train_dataset = {}
    for lecture in training_set:
        input_filename = features_dir + "/" + database.name + "_" + lecture.title + ".pickle"

        train_dataset[lecture.title.lower()] = MiscHelper.dump_load(input_filename)

    # prepare the cross validation
    complete_conf_matrix = {target: {pred: 0 for pred in unique_label} for target in unique_label}
    train_all_acc = []

    # cross-validate ....
    for lecture in training_set:
        train_list = training_titles.copy()
        test_list = [lecture.title.lower()]
        train_list.remove(lecture.title.lower())

        train_x, train_y, train_frame_infos = PoseFeatureExtractor.combine_datasets(train_list, train_dataset)
        test_x, test_y, test_frame_infos = PoseFeatureExtractor.combine_datasets(test_list, train_dataset)

        # train classifier and compute confusion matrix
        clf = RandomForestClassifier(n_estimators=64, max_depth=16, random_state=0)
        clf = clf.fit(train_x, train_y)

        pred_train_y = clf.predict(train_x)
        train_acc = (train_y == pred_train_y).sum() / len(train_y)
        train_all_acc.append(train_acc)
        print("Training Accuracy is {0}".format(train_acc))

        y_pred = clf.predict(test_x)
        y_pred_copy = y_pred.reshape((y_pred.shape[0], 1))
        test_name_dup = test_list * len(test_y)
        all_values = test_name_dup, test_frame_infos, test_y, y_pred_copy.tolist()

        spreadsheet.record_results(all_values)

        # get the label probabilities
        all_classes = clf.classes_
        y_prob = clf.predict_proba(test_x)

        infos = np.concatenate((y_pred_copy, y_prob), axis=1)

        # save the label_prob
        csv_col_prob = csv_col[1:] + all_classes.tolist()
        all_values = test_frame_infos, test_y, infos.tolist()

        label_prob_filename = label_probabilities_dir + "/" + database.name + "_" + lecture.title + ".csv"
        label_probs_output = ResultRecorder(label_prob_filename)
        label_probs_output.write_headers(csv_col_prob)
        label_probs_output.record_results(all_values)

        local_conf_matrix = {target: {pred: 0 for pred in unique_label} for target in unique_label}
        for idx in range(len(y_pred)):
            complete_conf_matrix[test_y[idx]][y_pred[idx]] += 1
            local_conf_matrix[test_y[idx]][y_pred[idx]] += 1
            # here
        print("Showing the local confusion matrix...")
        test_acc = print_confusion_matrix(unique_label, local_conf_matrix, lecture.title)


        print("test ready")


    print("Showing the complete confusion matrix...")
    test_acc_avg = print_confusion_matrix(unique_label, complete_conf_matrix, "complete")

    train_acc_avg = np.mean(np.array(train_all_acc))
    print("Training Average Accuracy is {0:.4f}".format(train_acc_avg * 100))
    print("Testing Average Accuracy is {0:.4f}".format(test_acc_avg))


if __name__ == '__main__':
    main()

