# AccessMath_Pose
Code and Data for ICDAR 2019 paper: Content Extraction from Lecture Video via Speaker Action Classification based on Pose Information

## Code - Overview

Export Frames(required for annotation tools)
------
Command: 
> python pre_ST3D_v2.0_00_export_frames.py [config] [mode] [parameters]  

It exports frames from original videos for Video annotation. The FPS of the original video is 30 and we export 10 frames per second to ensure the annotator work correctly.

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
Command:
> python gt_annotator.py [config] [lecture_name]

This annotator is used to label the intervels of speaker action. For each interval, the lable contains the beginning and ending frame number of the interval, and the action of the speaker during the interval. We release the annotation data for this paper. More information about the annotator could be accessed from (________________)

Examples:

For one specific lecture:
> python gt_annotator.py configs\02_labeling.conf lecture_01


Running Openpose 
------

openpose_00_run.py
openpose_01_combine.py

Training
------
#### Get Action Segment Information
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
