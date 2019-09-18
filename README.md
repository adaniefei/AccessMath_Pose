# AccessMath_Pose
Code and Data for ICDAR 2019 paper: Content Extraction from Lecture Video via Speaker Action Classification based on Pose Information

Export Frames
------
#### Export Frames from Video(required for annotation tools)
  python pre_ST3D_v2.0_00_export_frames.py [config] [mode] [parameters]
  
  
  Examples:
    For all lectures in AccessMath(including the ones not currently in used):
      > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf
    For all lectures from more than one dataset with one command:
      > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf -d "training testing"
    For one specific lecture:
      > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf -l lecture_01
    Similarly, for a set of lectures: 
      > python pre_ST3D_v2.0_00_export_frames.py configs\01_export_frames.conf -l "lecture_01 lecture_02 lecture_06"
       
  Where
    |config | configs\01_export_frames.conf|
    |lecture_name | video name from *Video List*|
  
Video Annotation
------
gt_annotator.py

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
