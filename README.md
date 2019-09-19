# AccessMath_Pose
Code and Data for ICDAR 2019 paper: Content Extraction from Lecture Video via Speaker Action Classification based on Pose Information

## Data
The data for this paper is released as [AccessMath_ICDAR_2019_data.zip](https://www.dropbox.com/s/5tk5zi5aytyf7ni/AccessMath_ICDAR_2019_data.zip?dl=0). Please download and unzip the file, and copy *data* folder to the code root directory *AccessMath_Pose*.


## Code - Overview
![alt text](https://github.com/adaniefei/Other/blob/images/system_arch.png?raw=true "Logo Title Text 1")
Export Frames(required for annotation tools)
------
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
       

  
Video Annotation
------
![alt text](https://github.com/adaniefei/Other/blob/images/gt_annotator.png?raw=true "Logo Title Text 1")

This annotator is used to label the intervels of speaker action and export the annotations. For each interval, the lable contains the beginning and ending frame number of the interval, and the action of the speaker during the interval. The annotation data for this paper is in *data\output\annotations*. More information about the annotator could be accessed from (________________)

       Command:
       > python gt_annotator.py [config] [lecture_name]

       Examples:
       For one specific lecture:
       > python gt_annotator.py configs\02_labeling.conf lecture_01


Running Openpose 
------
Two scripts are used for getting OpenPose data. 

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
The user needs to download the original videos from [AccessMath](https://www.cs.rit.edu/~accessmath/am_videos/) and re-encode them to MP4 format. In order to use the other script later, each lecture should be re-encoded into a single MP4 video.


2. *openpose_01_combine.py* is used to combine the data from all json files of one lecture video into one single csv file. It also manages the special case when there is no human subject in the screen or OpenPose fails to capture the keypoints. In this case a json file doesn't contain the speaker information and set all keypoints information with an invalid value(e.g. -1000). We provide the output csv files of this script in *data\output\openpose_csv*.

       Commands:
       For all training videos
       > python openpose_01_combine.py configs\03_main.conf -d training

       For all testing videos
       > python openpose_01_combine.py configs\03_main.conf -d testing


Training
------
#### Get Action Segment Information
*spk_train_00_get_action_segments.py* is used to get the action segment information from the annotation output of training videos. It uses
   python spk_train_00_get_action_segments.py configs\03_main.conf
   python spk_train_01_segment_pose_data.py configs\03_main.conf
   python spk_train_02_get_features.py configs\03_main.conf
   python spk_train_03_train_classifier.py configs\03_main.conf
   python spk_train_04_crossvalidation.py configs\03_main.conf

   python train_ml_binarizer.py configs\03_main.conf
   
  Usage
    python spk_train_00_get_action_segments.py [config]
  Where
    config - > configs\03_main.conf
    
#### Extract
  
spk_train_01_segment_pose_data.py
spk_train_02_get_features.py
spk_train_03_train_classifier.py
spk_train_04_crossvalidation.py

Summarization
------
spk_summ_00_segment.py
spk_summ_01_get_features.py
spk_summ_02_classify_actions.py
spk_summ_03_get_bboxes_per_frame.py
spk_summ_04_extract_video_metadata.py
spk_summ_05_temporal_segmentation.py
spk_summ_06_keyframe_extraction.py
spk_summ_07_fg_estimation.py
spk_summ_08_generate_summaries.py
eval_multiple_summaries.py
