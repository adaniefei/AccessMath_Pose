
import os
import sys

from AccessMath.preprocessing.video_worker.frame_exporter import FrameExporter
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
# from AccessMath.preprocessing.config.parameters import Parameters

def create_VOC_dirs(lecture_title, frame_export_dir):
    lecture_dir = frame_export_dir + "/" + lecture_title + '/'
    voc_subdirs = ['Annotations', 'ImageSets', 'JPEGImages']
    for voc_subdir in voc_subdirs:
        export_path = lecture_dir + voc_subdir
        if not os.path.isdir(export_path):
            os.makedirs(export_path)

def get_worker(process):
    frame_export_dir = process.configuration.get("OUTPUT_FRAME_EXPORT")
    frame_export_format = process.configuration.get("OUTPUT_FRAME_EXPORT_FORMAT")

    frame_export_quality = process.configuration.get_int("OUTPUT_FRAME_EXPORT_QUALITY", 100)
    
    create_VOC_dirs(process.current_lecture.title, frame_export_dir)
    
    export_dir = frame_export_dir + "/" + process.current_lecture.title + "/JPEGImages"
    
    frame_exporter = FrameExporter(export_dir, img_extension=frame_export_format, img_quality=frame_export_quality)
    return frame_exporter

def main():
    #usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
        return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], None, None)
    if not process.initialize():
        return

    
    fps = process.configuration.get_float("FRAME_EXPORT_FPS")
    process.start_video_processing(fps, get_worker, None, 0, True, True)

    print("Finished")


if __name__ == "__main__":
    main()

