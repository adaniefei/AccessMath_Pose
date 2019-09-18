

import os
from subprocess import call
import datetime
import sys
import argparse

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess

def video_process(process, input_data):

    if len(process.current_lecture.main_videos) > 1:
        print("Warning: There are multiple video files for current lecture, only the first will be processed")

    vid_abs = process.configuration.get_str("VIDEO_FILES_PATH") + "/" + process.current_lecture.main_videos[0]["path"]

    lecture_prefix = process.database.name + "_" + process.current_lecture.title
    odir = (process.configuration.get_str("OUTPUT_PATH") + "/" +
            process.configuration.get_str("OPENPOSE_OUTPUT_DIR_JSON") + "/" + lecture_prefix)

    odir_json = odir + "/" + 'json'

    # Flags
    gesture = process.configuration.get_bool("OPENPOSE_GESTURE")
    render_frame = process.configuration.get_bool("OPENPOSE_RENDER_FRAME")
    render_video = process.configuration.get_bool("OPENPOSE_RENDER_VIDEO")

    # create directory corresponding to the video name
    os.makedirs(odir_json, exist_ok=True)

    # We are assuming that OpenPoseDemo is on PATH,
    # otherwise replace this with the absolute path to OpenPoseDemo executable.
    runfile = process.configuration.get_str("OPENPOSE_DEMO_PATH")

    command = [runfile, "--video", vid_abs, "--write_json", odir_json]
    # display: no, render_pose: yes, face landmark: yes, hand skeleton: yes, maximum traking people: 1
    command += ["--display", "0", "--render_pose", "1", "--number_people_max", "1"]
    # if extract gesture feature
    if gesture:
        # render hand with gpu
        command +=["--hand", "--hand_render", "2"]
    # if render frames
    if render_frame:
        odir_frame = odir + "/" + 'frame'
        os.makedirs(odir_frame, exist_ok=True)
        command += ["--write_images", odir_frame, "--write_images_format jpg"]

    # if render video
    if render_video:
        odir_video = odir + "/" + lecture_prefix + ".mp4"
        command += ["--write_video", odir_video]

    # print(command)

    print('Skeleton and Rendered Start! -> ' + str(datetime.datetime.utcnow()))
    print(command)
    call(command)
    print('Skeleton and Rendered Completed! -> ' + str(datetime.datetime.utcnow()))
    return


def main(argv):
    # usage check
    if not ConsoleUIProcess.usage_with_config_check(sys.argv):
        return

    process = ConsoleUIProcess.FromConfigPath(sys.argv[1], sys.argv[2:], None, None)
    if not process.initialize():
        return

    process.start_input_processing(video_process)


if __name__ == '__main__':
    main(sys.argv)

