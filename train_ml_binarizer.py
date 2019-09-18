
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from AM_CommonTools.configuration.configuration import Configuration
from AM_CommonTools.util.time_helper import TimeHelper

from AccessMath.annotation.keyframe_annotation import KeyFrameAnnotation
from AccessMath.annotation.unique_cc_group import UniqueCCGroup
from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.evaluation.eval_parameters import EvalParameters
from AccessMath.evaluation.evaluator import Evaluator
from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.content.MLBinarizer import MLBinarizer
from AccessMath.preprocessing.tools.patch_sampling import PatchSampling


# TODO: refactor this functionality into the main framework
def load_keyframes(root_dir, database):
    all_keyframes = []
    binarized_keyframes = []

    training_set = database.get_dataset("training")
    for lecture in training_set:
        print(lecture.title.lower())
        annotation_prefix = root_dir + "/" + database.output_annotations + "/" + database.name + "_" + lecture.title.lower()

        annot_filename = annotation_prefix + "/segments.xml"
        annot_image_prefix = annotation_prefix + "/keyframes/"
        annot_binary_prefix = annotation_prefix + "/binary/"

        # ideal summary ...
        annot_keyframes = KeyFrameAnnotation.LoadExportedKeyframes(annot_filename, annot_image_prefix, False, False)
        all_keyframes += annot_keyframes

        for kf in annot_keyframes:
            kf.binary_image = cv2.imread(annot_binary_prefix + str(kf.idx) + ".png")
            kf.update_binary_cc(False)

            bin_kf = KeyFrameAnnotation(kf.database, kf.lecture, kf.idx, kf.time, kf.objects, kf.raw_image)
            binarized_keyframes.append(bin_kf)

    return all_keyframes, binarized_keyframes

def generate_fake_keyframe_info(all_keyframes):
    # here, we use fake unique CC's as we care to improve per-frame recall/precision
    fake_unique_groups = []
    fake_cc_group = []
    fake_segments = []
    for kf_idx, keyframe in enumerate(all_keyframes):
        fake_segments.append((kf_idx * 5 + 1, kf_idx * 5 + 4))
        fake_cc_group.append({})

        for cc in keyframe.binary_cc:
            new_group = UniqueCCGroup(cc, kf_idx)
            fake_unique_groups.append(new_group)
            fake_cc_group[kf_idx][cc.strID()] = new_group

    return fake_unique_groups, fake_cc_group, fake_segments


def get_patch_features_raw_values(patch):
    # use raw edge values
    return patch.reshape(patch.size).tolist()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("\tpython train_ml_binarizer.py config [force_update] [classifier_file] [patch_size]")
        print("")
        print("Where")
        print("\tconfig\t\t\tPath to config file")
        print("\tforce_update \t\tOptional, force to update the sampled Patch file")
        print("\tclassifier_file \tOptional, force classifier path diff. from Config")
        print("\tpatch_size \t\tOptional, override patch size")
        return

    # read the configuration file ....
    config = Configuration.from_file(sys.argv[1])

    # load the database
    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid database file")
        return

    # <Parameters>
    run_crossvalidation = config.get("ML_BINARIZER_TRAIN_RUN_CROSSVALIDATION", True)

    if not config.contains("ML_BINARIZER_PATCHES_FILENAME"):
        print("Must specificy a file to store sampled patches")
        return

    output_dir = config.get_str("OUTPUT_PATH")
    ml_binarizer_dir = output_dir + "/" + config.get_str("ML_BINARIZER_DIR")
    patch_filename = ml_binarizer_dir + "/" + config.get_str("ML_BINARIZER_PATCHES_FILENAME")

    # For debugging/comparsion, use OTSU binarization
    OTSU_mode = config.get("ML_BINARIZER_TRAIN_OTSU_MODE", False)

    # baseline_mode = True # Train Random Forest instead .
    retrain_classifier = config.get("ML_BINARIZER_TRAIN_RETRAIN", True)

    if not config.get("ML_BINARIZER_OVERRIDE_PARAMETERS", False):
        # Sampling mode #1: Distribution of proportions
        #
        # of 100% pixels, we sample fg_proportion from GT Foreground pixels (handwriting pixels)
        #    handwriting pixels = fg_proportion
        #    all background     = (1 - fg_proportion)
        #
        # The remaining background pixels are sampled as close or far from foreground
        #    Close to Foreground pixels = (1 - fg_proportion) * bg_close_prop
        #    Remaining background pixels = (1 - fg_proportion) * (1 - bg_close_prop)
        #
        # The last proportion of pixels can be obtained from whiteboard or background objects, we separate them as
        #    Not Close Whiteboard background pixels = (1 - fg_proportion) * (1 - bg_close_prop) * bg_board_prop
        #    Not Whiteboard background pixels       = (1 - fg_proportion) * (1 - bg_close_prop) * (1 - bg_board_prop)
        #
        # Sampling mode #2: Distribution of proportions
        #
        # of 100% pixels, we sample fg_proportion from GT Foreground pixels (handwriting pixels)
        #    handwriting pixels = fg_proportion
        #    all background     = (1 - fg_proportion)
        #
        # The remaining background pixels by average intensity of the window, the greater the average, the more likely they
        #   are to be sampled. This is consistent with sampling mode 1, but is less discrete and requires less parameters
        sampling_mode = Parameters.MLBin_sampling_mode

        patch_size = Parameters.MLBin_patch_size
        patches_per_frame = Parameters.MLBin_sampling_patches_per_frame
        fg_proportion = Parameters.MLBin_sampling_fg_proportion
        bg_close_prop = Parameters.MLBin_sampling_bg_close_prop
        bg_board_prop = Parameters.MLBin_sampling_bg_board_prop

        mlbin_sigma_color = Parameters.MLBin_sigma_color
        mlbin_sigma_space = Parameters.MLBin_sigma_space
        mlbin_median_blur_k = Parameters.MLBin_median_blur_k
        mlbin_dark_background = Parameters.MLBin_dark_background

        feature_workers = Parameters.MLBin_train_workers

        # Random Forest
        rf_n_trees = Parameters.MLBin_rf_n_trees           # 16
        rf_max_depth = Parameters.MLBin_rf_max_depth       # 12
        rf_max_features = Parameters.MLBin_rf_max_features # 32

    else:
        print("Reading ML Binarizer parameters from config ...")

        sampling_mode = config.get_int("ML_BINARIZER_SAMPLING_MODE", 2)

        patch_size = config.get_int("ML_BINARIZER_PATCH_SIZE", 7)
        patches_per_frame = config.get_int("ML_BINARIZER_SAMPLING_PATCHES_PER_FRAME", 20000)
        fg_proportion = config.get_float("ML_BINARIZER_SAMPLING_FG_PROPORTION", 0.5)
        bg_close_prop = config.get_float("ML_BINARIZER_SAMPLING_BG_CLOSE_PROPORTION", 0.9)
        bg_board_prop = config.get_float("ML_BINARIZER_SAMPLING_BG_BOARD_PROPORTION", 1.0)

        mlbin_sigma_color = config.get_float("ML_BINARIZER_SIGMA_COLOR", 13.5)
        mlbin_sigma_space = config.get_float("ML_BINARIZER_SIGMA_SPACE", 4.0)
        mlbin_median_blur_k = config.get_int("ML_BINARIZER_MEDIAN_BLUR_K", 33)
        mlbin_dark_background = config.get("ML_BINARIZER_DARK_BACKGROUND")

        feature_workers = config.get_int("ML_BINARIZER_TRAIN_WORKERS", 7)

        # Random Forest
        rf_n_trees = config.get_int("ML_BINARIZER_RF_N_TREES", 16)  # 16
        rf_max_depth = config.get_int("ML_BINARIZER_RF_MAX_DEPTH", 12)  # 12
        rf_max_features = config.get_int("ML_BINARIZER_RF_MAX_FEATURES", 32) # 32


    if len(sys.argv) >= 4:
        # user specified location
        classifier_file = sys.argv[3]
    else:
        # by default, store at the place specified in the configuration or parameters file ...
        if not config.get("ML_BINARIZER_OVERRIDE_PARAMETERS", False):
            classifier_file = Parameters.MLBin_classifier_file
        else:
            classifier_file = ml_binarizer_dir + "/" + config.get_str("ML_BINARIZER_CLASSIFIER_FILENAME")

    feature_function = get_patch_features_raw_values

    # </Parameters>

    if len(sys.argv) >= 3:
        try:
            force_update = int(sys.argv[2]) > 0
        except:
            print("Invalid value for force_udpate")
            return
    else:
        force_update = False

    if len(sys.argv) >= 5:
        try:
            patch_size = int(sys.argv[4])
        except:
            print("Invalid value for patch_size")
            return

    assert (patch_size - 1) % 2 == 0
    bg_close_neighborhood = int((patch_size - 1) / 2) + 1
    
    print("Classifier Path: " + classifier_file)
    ml_binarizer = MLBinarizer(None, patch_size, mlbin_sigma_color, mlbin_sigma_space, mlbin_median_blur_k,
                               mlbin_dark_background)

    print("... loading data ...")
    start_loading = time.time()
    all_keyframes, binarized_keyframes = load_keyframes(output_dir, database)
    fake_unique_groups, fake_cc_group, fake_segments = generate_fake_keyframe_info(all_keyframes)

    print("Total Training keyframes: " + str(len(all_keyframes)))

    end_loading = time.time()
    start_preprocessing = time.time()

    print("Pre-processing key-frames", flush=True)
    all_preprocessed = []
    for kf_idx, kf in enumerate(all_keyframes):
        all_preprocessed.append(ml_binarizer.preprocessing(kf.raw_image))
        # cv2.imwrite("DELETE_NOW_tempo_bin_input_" + str(kf_idx) + ".png", all_preprocessed[-1])

    end_preprocessing = time.time()
    start_patch_extraction = time.time()

    # Extracting/Loading patches used for training (only if not on OTSU's mode)
    if not OTSU_mode:
        # generate the patch-based training set ...
        # check if patch file exists ...
        if not os.path.exists(patch_filename) or force_update:
            print("Extracting patches...")

            if sampling_mode == 1:
                # SampleEdgeFixBg()
                patches = PatchSampling.SampleEdgeFixBg(all_keyframes, all_preprocessed, patch_size, patches_per_frame,
                                                        fg_proportion, bg_close_prop, bg_board_prop, bg_close_neighborhood)
            elif sampling_mode == 2:
                # SampleEdgeContBg
                patches = PatchSampling.SampleEdgeContBg(all_keyframes, all_preprocessed, patch_size, patches_per_frame,
                                                         fg_proportion)
            else:
                patches = (None, None)

            patches_images, patches_labels = patches

            # generate features
            print("\nGenerating features ...", flush=True)
            all_features = []
            with ProcessPoolExecutor(max_workers=feature_workers) as executor:
                for lect_idx, lecture_images in enumerate(patches_images):
                    print("Processing patches from lecture {0:d} out of {1:d}".format(lect_idx + 1, len(patches_images)))
                    lecture_features = []
                    for i, patch_features in enumerate(executor.map(feature_function, lecture_images)):
                        lecture_features.append(patch_features)

                    all_features.append(lecture_features)

            print("\nSaving patches and features to file")
            out_file = open(patch_filename, "wb")
            pickle.dump(patches_labels, out_file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(patches_images, out_file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_features, out_file, pickle.HIGHEST_PROTOCOL)
            out_file.close()
        else:
            # load patches from file ....
            print("Loading patches and features from file")
            in_file = open(patch_filename, "rb")
            patches_labels = pickle.load(in_file)
            patches_images = pickle.load(in_file)
            all_features = pickle.load(in_file)
            in_file.close()

    end_patch_extraction = time.time()
    
    total_training_time = 0.0
    total_binarization_time = 0.0
    total_evaluation_time = 0.0

    cross_validated_classifiers = []
    
    if not OTSU_mode:
        start_training = time.time()

        # train classifier using training patches ...
        count_all_patches = sum([len(lecture_images) for lecture_images in patches_images])
        print("Total patches available for training: " + str(count_all_patches))

        n_features = len(all_features[0][0])
        print("Total Features: " + str(n_features))

        # check local performance using cross-validation based on leaving one lecture out
        conf_matrix = np.zeros((2, 2), dtype=np.int32)
        avg_train_accuracy = 0.0
        rf_max_features = min(rf_max_features, n_features)

        if run_crossvalidation:
            for i in range(len(patches_images)):
                print("Cross-validation fold #" + str(i + 1))

                training_data = []
                training_labels = []
                testing_data = []
                testing_labels = []
                for k in range(len(patches_images)):
                    if i == k:
                        testing_data += all_features[k]
                        testing_labels += patches_labels[k]
                    else:
                        training_data += all_features[k]
                        training_labels += patches_labels[k]

                training_data = np.array(training_data)
                testing_data = np.array(testing_data)

                print("-> Training Samples: " + str(training_data.shape[0]))
                print("-> Testing Samples: " + str(testing_data.shape[0]))

                # classification mode ...
                # random forest ...
                classifier = RandomForestClassifier(rf_n_trees, max_features=rf_max_features, max_depth=rf_max_depth, n_jobs=-1)
                classifier.fit(training_data, training_labels)
                
                # keep reference to the n-th fold classifier
                cross_validated_classifiers.append(classifier)

                pred_labels = classifier.predict(training_data)
                train_conf_matrix = np.zeros((2, 2), dtype=np.int32)
                for train_idx in range(len(training_labels)):
                    train_conf_matrix[training_labels[train_idx], pred_labels[train_idx]] += 1
                pixel_accuracy = (train_conf_matrix[0, 0] + train_conf_matrix[1, 1]) / len(training_labels)
                print("-> Train pixel accuracy: " + str(pixel_accuracy * 100.0))
                avg_train_accuracy += pixel_accuracy

                pred_labels = classifier.predict(testing_data)

                for test_idx in range(len(testing_labels)):
                    conf_matrix[testing_labels[test_idx], pred_labels[test_idx]] += 1


            pixel_accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / count_all_patches
            avg_train_accuracy /= len(all_features)

            print("Combined testing confusion matrix: ")
            print(conf_matrix)
            print("Final training pixel accuracy: " + str(avg_train_accuracy * 100.0))
            print("Final testing pixel accuracy: " + str(pixel_accuracy * 100.0))

        # now, use all data to train a classifier for binarization of all frames ...
        if not os.path.exists(classifier_file) or force_update or retrain_classifier:
            print("Training classifier using all patches", flush=True)
            # classification
            training_data = []
            training_labels = []
            for k in range(len(patches_images)):
                training_data += all_features[k]
                training_labels += patches_labels[k]

            training_data = np.array(training_data)

            # Train Random Forest
            classifier = RandomForestClassifier(rf_n_trees, max_features=rf_max_features, max_depth=rf_max_depth, n_jobs=-1)
            classifier.fit(training_data, training_labels)
            
            print("Saving classifier to file")
            out_file = open(classifier_file, "wb")
            pickle.dump(classifier, out_file, pickle.HIGHEST_PROTOCOL)
            out_file.close()
        else:
            print("Loading classifier from file")
            in_file = open(classifier_file, "rb")
            classifier = pickle.load(in_file)
            in_file.close()

        # release memory (a lot) of elements that will not be used after this point ...
        all_features = None
        patches_labels = None
        training_data = None
        training_labels = None
        testing_data = None
        testing_labels = None

        end_training = time.time()
        total_training_time += end_training - start_training
    
    # binarize using parameter combination...
    start_binarizing = time.time()

    last_lecture = None
    lecture_offset = -1
    training_set = database.get_dataset("training")
    
    for idx, bin_kf in enumerate(binarized_keyframes):
        if bin_kf.lecture != last_lecture:
            last_lecture = bin_kf.lecture
            lecture_offset += 1

        print("binarizing kf #" + str(idx) + ", from " + training_set[lecture_offset].title, end="\r", flush=True)

        if OTSU_mode:
            # ideal BG removal ...
            #strel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(patch_size), int(patch_size)))
            #bg_mask = all_keyframes[idx].object_mask > 0
            #all_preprocessed[idx][bg_mask] = 0

            otsu_t, bin_res = cv2.threshold(all_preprocessed[idx].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            bin_kf.binary_image = np.zeros((bin_res.shape[0], bin_res.shape[1], 3), dtype=np.uint8)
            bin_kf.binary_image[:, :, 0] = 255 - bin_res.copy()
            bin_kf.binary_image[:, :, 1] = bin_kf.binary_image[:, :, 0].copy()
            bin_kf.binary_image[:, :, 2] = bin_kf.binary_image[:, :, 0].copy()
        else:
            # set classifier for binarization ....
            if run_crossvalidation:
                # use the classifier that has not seen this image ...
                ml_binarizer.classifier = cross_validated_classifiers[lecture_offset]
            else:
                # use the globally train classifier
                ml_binarizer.classifier = classifier

            # ... binarize the pre-processed image ... 
            binary_image = ml_binarizer.preprocessed_binarize(all_preprocessed[idx])
            
            # Do hystheresis filtering ...
            otsu_t, high_bin = cv2.threshold(all_preprocessed[idx].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            low_bin = binary_image
            
            filtered_bin = 255 - MLBinarizer.binary_hysteresis(low_bin, high_bin)
            
            bin_kf.binary_image = np.zeros((filtered_bin.shape[0], filtered_bin.shape[1], 3), dtype=np.uint8)
            bin_kf.binary_image[:, :, 0] = filtered_bin
            bin_kf.binary_image[:, :, 1] = filtered_bin
            bin_kf.binary_image[:, :, 2] = filtered_bin

        bin_kf.update_binary_cc(False)

        if config.get("ML_BINARIZER_SAVE_BINARY", True):
            if OTSU_mode:
                out_name = "TEMPO_OTSU_baseline_binarized_" + str(idx) + ".png"
            else:
                out_name = "TEMPO_rf_baseline_binarized_" + str(idx) + ".png"

            cv2.imwrite(out_name, bin_kf.binary_image)
        
    end_binarizing = time.time()
    total_binarization_time += end_binarizing - start_binarizing

    # run evaluation metrics ...
    print("Computing final evaluation metrics....")
    
    # Summary level metrics ....
    start_evaluation = time.time()
    
    EvalParameters.UniqueCC_global_tran_window = 1
    EvalParameters.UniqueCC_min_precision = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95]
    EvalParameters.UniqueCC_min_recall = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95]
    EvalParameters.Report_Summary_Show_Counts = False
    EvalParameters.Report_Summary_Show_AVG_per_frame = False
    EvalParameters.Report_Summary_Show_Globals = True

    all_scope_metrics, scopes = Evaluator.compute_summary_metrics(fake_segments, all_keyframes, fake_unique_groups,
                                                            fake_cc_group, fake_segments, binarized_keyframes,
                                                            False)


    for scope in scopes:
        print("")
        print("Metrics for scope: " + scope)
        print("      \t      \tRecall\t      \t       \tPrecision")
        print("Min R.\tMin P.\tE + P\tE. Only\tP. Only\tE + P\tE. Only\tP. Only\tBG. %\tNo BG P.")
        scope_metrics = all_scope_metrics[scope]

        recall_percent_row = "{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}"
        prec_percent_row = "{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}"

        for all_metrics in scope_metrics:
            metrics = all_metrics["recall_metrics"]

            recall_str = recall_percent_row.format(all_metrics["min_cc_recall"] * 100.0,
                                                   all_metrics["min_cc_precision"] * 100.0,
                                                   metrics["recall"] * 100.0, metrics["only_exact_recall"] * 100.0,
                                                   metrics["only_partial_recall"] * 100.0)

            metrics = all_metrics["precision_metrics"]

            prec_str = prec_percent_row.format(metrics["precision"] * 100.0, metrics["only_exact_precision"] * 100.0,
                                               metrics["only_partial_precision"] * 100.0,
                                               metrics["global_bg_unmatched"] * 100.0,
                                               metrics["no_bg_precision"] * 100.0)

            print(recall_str + "\t" + prec_str)

    # pixel level metrics
    pixel_metrics = Evaluator.compute_pixel_binary_metrics(all_keyframes, binarized_keyframes)
    print("Pixel level metrics")
    for key in sorted(pixel_metrics.keys()):
        print("{0:s}\t{1:.2f}".format(key, pixel_metrics[key] *100.0))
    
    end_evaluation = time.time()
    total_evaluation_time += end_evaluation - start_evaluation
    end_everything = time.time()

    print("Total loading time: " + TimeHelper.secondsToStr(end_loading - start_loading))
    print("Total preprocessing time: " + TimeHelper.secondsToStr(end_preprocessing - start_preprocessing))
    print("Total patch extraction time: " + TimeHelper.secondsToStr(end_patch_extraction - start_patch_extraction))
    print("Total training time: " + TimeHelper.secondsToStr(total_training_time))
    print("Total binarization time: " + TimeHelper.secondsToStr(total_binarization_time))
    print("Total evaluation time: " + TimeHelper.secondsToStr(total_evaluation_time))
    print("Total Time: " + TimeHelper.secondsToStr(end_everything - start_loading))
    


if __name__ == '__main__':
    main()
