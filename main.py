import os
from query import *
from frameextract import extract_video_frames
ego4d_path = os.listdir("/home/sebastian/extssd/ego4d/v2/full_scale")

def main(video_directory, frame_directory, model, tokenizer, prompt_list, results_directory):
    video_name_list = os.listdir(video_directory)
    for video in video_name_list:
        video_path = os.path.join(video_directory, video)
        print(video_path)

