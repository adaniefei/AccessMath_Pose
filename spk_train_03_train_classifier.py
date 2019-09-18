
import os
import sys

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.speaker.actions.pose_feature_extractor import PoseFeatureExtractor

from sklearn.ensemble import RandomForestClassifier

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
    classifier_dir = output_dir + "/" + config.get_str("SPEAKER_ACTION_CLASSIFIER_DIR")
    os.makedirs(classifier_dir, exist_ok=True)
    classifier_filename = classifier_dir + "/" + config.get_str("SPEAKER_ACTION_CLASSIFIER_FILENAME")

    dataset_name = config.get("SPEAKER_TRAINING_SET_NAME")
    training_set = database.datasets[dataset_name]
    training_titles = [lecture.title.lower() for lecture in training_set]

    # get classifier parameters
    rf_n_trees = config.get_int("SPEAKER_ACTION_CLASSIFIER_RF_TREES", 64)
    rf_depth = config.get_int("SPEAKER_ACTION_CLASSIFIER_RF_DEPTH", 16)

    # read all training data available ....
    train_dataset = {}
    for lecture in training_set:
        input_filename = features_dir + "/" + database.name + "_" + lecture.title + ".pickle"
        train_dataset[lecture.title.lower()] = MiscHelper.dump_load(input_filename)

    train_x, train_y, train_frame_infos = PoseFeatureExtractor.combine_datasets(training_titles, train_dataset)

    # classify and confusion matrix part
    clf = RandomForestClassifier(n_estimators=rf_n_trees, max_depth=rf_depth, random_state=0)
    clf = clf.fit(train_x, train_y)

    MiscHelper.dump_save(clf, classifier_filename)


if __name__ == '__main__':
    main()

