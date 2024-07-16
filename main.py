from frameextract import extract_video_frames
from query import *

def process(video_directory, frame_directory, results_directory):
    video_list = os.listdir(video_directory)
    for video in video_list:
        video_path = os.path.join(video_directory, video)
        extract_video_frames(video_path, frame_directory, 1)
        process_frame_directory(frame_directory, model, tokenizer, prompt_list, results_directory)


def proces_selected(video_list, frame_directory, results_directory):
    for video in video_list:
        video_path = os.path.join(r"D:\ego4d\v2\full_scale", video)
        extract_video_frames(video_path, frame_directory, 1)
    process_frame_directory(frame_directory, model, tokenizer, prompt_list, results_directory)

video_list = ["2b012751-2e50-4acb-b645-c864930e92c8.mp4", "2b9a1020-d876-4a3c-9837-4c9ba560d9d4.mp4",
              "3a32c2ed-6988-40fe-a8e9-a2188f001f11.mp4", "6baf5a36-83e6-4fb4-81bc-970dffd1df49.mp4", "ab2e1e0e-a24c-4320-9479-646b0b46365e.mp4"]

proces_selected(video_list, 'indoorvids/indoorvidframes', 'indoorvids/indoorvidresults')