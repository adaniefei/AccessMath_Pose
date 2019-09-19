# AccessMath_Pose
Code and Data for ICDAR 2019 paper: Content Extraction from Lecture Video via Speaker Action Classification based on Pose Information

## Data
The data for this paper is released as [AccessMath_ICDAR_2019_data.zip](https://www.dropbox.com/s/5tk5zi5aytyf7ni/AccessMath_ICDAR_2019_data.zip?dl=0). Please download and unzip the file, and copy *data* folder to the code root directory *AccessMath_Pose*. 
The orignal AccessMath videos need to be downloaded and saved in *data\original_videos\lectures* from [AccessMath](https://www.cs.rit.edu/~accessmath/am_videos/lectures/). 


## Code
This tool is tested and intended for usage with **Python 3.6+**.

Main Library Requirements:
 - Pygame
 - OpenCV
 - Numpy 
 - Shapely
 - pandas
 
### C files
The c file *accessmath_lib.c* needs to be re-compile for non-windows users. We provide a valid version *accessmath_lib.o* of the file for users of Python 3.6+ 64-bits for windows user.
 
### Speaker Action Annotation
#### Export Frames(required for annotation tools)
It exports frames from original videos for Video annotation. The FPS of the original video is 30 and we export `FRAME_EXPORT_FPS` frames per second given in the [config] to ensure the annotator work correctly.

       Command: 
       > python pre_ST3D_v2.0_00_export_frames.py [config] [mode] [parameters]  

       Examples:
       For all lectures in AccessMath(including the ones not currently in used):
       > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf

       For all lectures from more than one dataset with one command:
       > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf -d "training testing"

       For one specific lecture:
       > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf -l lecture_01

       Similarly, for a set of lectures: 
       > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf -l "lecture_01 lecture_02 lecture_06"
       

  
#### Video Annotation
![alt text](https://github.com/adaniefei/Other/blob/images/gt_annotator.png?raw=true "Logo Title Text 1")

This annotator is used to label the intervels of speaker action and export the annotations. For each interval, the lable contains the beginning and ending frame number of the interval, and the action of the speaker during the interval. The annotation data for this paper is in *data\output\annotations*. 

We adapt the annotator from the paper of *K. Davila, R. Zanibbi "Whiteboard video summarization via spatio-temporal conflict minimization", ICDAR 2017*. ([davila2017whiteboard](https://www.cs.rit.edu/~rlaz/files/Kenny_ICDAR_2017.pdf)). For our paper we add more features to complete the speaker action annotation.

       Command:
       > python gt_annotator.py [config] [lecture_name]

       Examples:
       For one specific lecture:
       > python gt_annotator.py configs\02_labeling.conf lecture_01

------

### Main Pipeline
![alt text](https://github.com/adaniefei/Other/blob/images/system_arch.png?raw=true "Logo Title Text 1")

#### Running Openpose 
Two scripts are used to get OpenPose data for **speaker pose estimation**. 

1. *openpose_00_run.py* is used to captured the keypoints of the speaker frame by frame from the lecture video. The output for each frame is a json file which includes the locations and detection confidence of all keypoints of the speaker. The json files will be saved in *data\output\openpose_json*.

       Commands:
       For all training videos
       > python openpose_00_run.py configs\03_main.conf -d training

       For all testing videos
       > python openpose_00_run.py configs\03_main.conf -d testing

To use *openpose_00_run.py*, the user must confirm that:

* OpenPose is installed and the Demo works successfully.
We use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect the keypoints of the speaker's body and hands. To install and quick start OpenPose Demo, please refer [Installation and Quick Start](https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation-reinstallation-and-uninstallation). 

* Copy the *models* folder from OpenPose directory to the code root directory *AccessMath_Pose*

* In the configure file *03_main.conf*, update the `OPENPOSE_DEMO_PATH` value to the corresponding *Demo* path.

* Copy the re-encoded versions of the original lecture videos into *data\mp4_videos*.
The user needs to re-encode the original AccessMath videos to MP4 format. In order to use the other script later, each lecture should be re-encoded into a single MP4 video.


2. *openpose_01_combine.py* is used to combine the data from all json files of one lecture video into one single csv file. It also manages the special case when there is no human subject in the screen or OpenPose fails to capture the keypoints. In this case a json file doesn't contain the speaker information and set all keypoints information with an invalid value(e.g. -1000). We provide the output csv files of this script in *data\output\openpose_csv*.

       Commands:
       For all training videos
       > python openpose_01_combine.py configs\03_main.conf -d training

       For all testing videos
       > python openpose_01_combine.py configs\03_main.conf -d testing


#### Speaker Motion Feature Extraction
##### For Training Video
1. *spk_train_00_get_action_segments.py* is used to get the action segment information from the annotation output of *training videos*. It uses the annotation of training videos and export action segment information in `SPEAKER_ACTION_SEGMENT_OUTPUT_DIR`.
`SPEAKER_ACTION_SEGMENT_LENGTH` `SPEAKER_ACTION_SEGMENT_SAMPLING_MODE` and `SPEAKER_ACTION_SEGMENT_SAMPLING_TRACKS` are used to generate the action segment in corresponding sampling setting.

       Command:
       > python spk_train_00_get_action_segments.py configs\03_main.conf
       
2.  *spk_train_01_segment_pose_data.py* generates the action segments of pose data given the action segment information from previous step and the speaker pose data of training videos captured from OpenPose. For each training video, the action segments of pose data and the normalization factor will be exported in `OUTPUT_PATH\SPEAKER_ACTION_SEGMENT_OUTPUT_DIR`.

        Command:
        > python spk_train_01_segment_pose_data.py configs\03_main.conf

3. *spk_train_02_get_features.py* extracts the selected features `SPEAKER_ACTION_FEATURE_POINTS ` from each action segment pose data, normalizes the features per training video and saves the features in `OUTPUT_PATH\SPEAKER_ACTION_FEATURES_DIR`

       Command:
       > python spk_train_02_get_features.py configs\03_main.conf

##### For Testing Video
1. *spk_summ_00_segment.py* generates the action segments of pose data seqentially given the segment length `SPEAKER_ACTION_SEGMENT_LENGTH` and the speaker pose data of testing videos captured from OpenPose. For each testing video, the action segments of pose data and the normalization factor will be exported in `OUTPUT_PATH\SPEAKER_ACTION_SEGMENT_OUTPUT_DIR`.

       Command:
       > python spk_summ_00_segment.py configs\03_main.conf
       
2. *spk_summ_01_get_features.py* extracts the selected features `SPEAKER_ACTION_FEATURE_POINTS` from each action segment pose data, normalizes the features per testing video and saves the features in `OUTPUT_PATH\SPEAKER_ACTION_FEATURES_DIR`

       Command:
       python spk_summ_01_get_features.py configs\03_main.conf
          
#### Speaker Action Classification
1. *spk_train_03_train_classifier.py* trains a Random Forest classifier by all training videos data. The classifier will be saved in `OUTPUT_PATH\SPEAKER_ACTION_CLASSIFIER_DIR\SPEAKER_ACTION_CLASSIFIER_FILENAME`.

       Command:
       > python spk_train_03_train_classifier.py configs\03_main.conf
       
2. *spk_train_04_crossvalidation.py* runs the 5-fold cross validation on training videos. It will print out the confusion matrix of each validation video, and the global confusion matrix in the end.

       Command:
       python spk_train_04_crossvalidation.py configs\03_main.conf
       
3. *spk_summ_02_classify_actions.py* uses the trained Random Forest classifier with the extracted features from each testing video. The result including beginning and ending frame number and the action prediction of each action segment will be saved in `OUTPUT_PATH\SPEAKER_ACTION_CLASSIFICATION_OUTPUT_DIR`. The classification probabilities detail required for later process will be saved in `SPEAKER_ACTION_CLASSIFICATION_PROBABILITIES_DIR`

       Command:
       python spk_summ_02_classify_actions.py configs\03_main.conf


#### Speaker Bounding Box Estimation
*spk_summ_03_get_bboxes_per_frame.py* generates the bounding box of speaker's body and writing hand per frame. `SPEAKER_IS_RIGHT_HANDED` manages the handedness of the speaker. Results will be saved in `SPEAKER_ACTION_CLASSIFICATION_BBOXES_DIR`.
     
     Command:
     python spk_summ_03_get_bboxes_per_frame.py configs\03_main.conf

#### Lecture Video Temporal Segmentation
1. *spk_summ_04_extract_video_metadata.py* extrats metadata(width and height) of every video and save the information in `SPEAKER_ACTION_VIDEO_META_DATA_DIR`.     

       Command:
       python spk_summ_04_extract_video_metadata.py configs\03_main.conf
     
    
2.  *spk_summ_05_temporal_segmentation.py* uses bounding box, video meta data and information in classification probability files to generate temporal segmentation of every videos. Those are saved in `SPEAKER_ACTION_TEMPORAL_SEGMENTS_DIR`.

        Command:
        python spk_summ_05_temporal_segmentation.py configs\03_main.conf


#### Foreground Mask Estimation
*spk_summ_07_fg_estimation.py* uses meta data and the bounding box information of every video to create the foreground mask based on writing action. `SPEAKER_FG_ESTIMATION_SPK_EXPANSION_FACTOR`, `SPEAKER_FG_ESTIMATION_MIN_MASK_FRAMES` and `SPEAKER_FG_ESTIMATION_MASK_EXPANSION_RADIUS` are used as mask expansion parameters. The result will be saved in `SPEAKER_FG_ESTIMATION_MASK_DIR`.

    Command:
    python spk_summ_07_fg_estimation.py configs\03_main.conf

#### Key-frame Selection and Binarization
1. *spk_summ_06_keyframe_extraction.py* selects all temporal keyframes of each lecture video by its temporal segmentation. Result will be saved in `SPEAKER_ACTION_KEYFRAMES_DIR`.

       Command:
       python spk_summ_06_keyframe_extraction.py configs\03_main.conf

2. *train_ml_binarizer.py* trains a binarizer classifier by the ideal keyframes from lecture videos and save it in `ML_BINARIZER_DIR`. We use the binarizer referring to *K. Davila, R. Zanibbi "Whiteboard video summarization via spatio-temporal conflict minimization", ICDAR 2017*. More detail of this paper could be found in [davila2017whiteboard](https://www.cs.rit.edu/~rlaz/files/Kenny_ICDAR_2017.pdf). 

       Command:
       python train_ml_binarizer.py configs\03_main.conf


#### Lecture Video Summarization and Evaluation
##### Generate Video Summarization
*spk_summ_08_generate_summaries.py* generates the final binarized summaries of lecture videos given the temporal segmentation, temporal keyframes and foreground mask. Summaries will be saved in `OUTPUT_PATH\summaries` 

    Command:
    python spk_summ_08_generate_summaries.py configs\03_main.conf

##### Evaluation
    
    Command:
    python eval_multiple_summaries.py configs\03_main.conf -d testing -b speaker_actions
   
The baseline parameter allows to evaluate variations of the system. During the generation of the summaries
the system uses the parameter "SPEAKER_SUMMARY_PREFIX" to add a baseline prefix to the generated summaries.
Here, we use the baseline parameter "-b speaker_actions" to allow the evaluation script find the right set of 
summaries to compute the corresponding evaluation metrics. 

Summaries from the ICDAR 2019 paper:

    python eval_multiple_summaries.py configs\03_main.conf -d testing -b speaker_actions_RAW
    python eval_multiple_summaries.py configs\03_main.conf -d testing -b speaker_actions_SPK_REMOVAL
    python eval_multiple_summaries.py configs\03_main.conf -d testing -b speaker_actions_BG_REMOVAL
    python eval_multiple_summaries.py configs\03_main.conf -d testing -b speaker_actions_SPK_BG_REMOVAL
    
