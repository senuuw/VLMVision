import cv2
import os
from PIL import Image

def extract_directory_5frames_raw(path, n):
    video_count = 0
    videos = os.listdir(path)[31:60] #start at 1 to ignore manifest)
    for video in videos:
        print(f'Starting Video {video}')
        timer_count = 0
        current_vid_path = os.path.join(r"D:\ego4d\v2\full_scale", video)
        cap = cv2.VideoCapture(current_vid_path)
        frame_count = 0
        video_count +=1
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                #try to open video, if error then skip video
                if ret:
                    if frame_count == 5:
                        break
                    else:
                        frame_count += 1
                        new_image_path = f'52scenes/{video[:-4]}_{frame_count:04d}.jpg'
                        cv2.imwrite(new_image_path, frame)
                        timer_count += 600 # i.e. at 30 fps, this advances one second, currently every 20s
                        cap.set(cv2.CAP_PROP_POS_FRAMES, timer_count)

                        # Check if image all black
                        new_img = Image.open(new_image_path)
                        if not new_img.getbbox():
                            os.remove(new_image_path)
                else:
                    cap.release()
                    break
            except:
                print(f'Video {video_count}, path:{current_vid_path} failed cap.read()')




def extract_video_frames(video_path, output_directory, frames_captured_per_second):
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Create a new directory with the video name
    video_output_directory = os.path.join(output_directory, video_name)
    os.makedirs(video_output_directory, exist_ok=True)

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timer_count = 0
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_path = os.path.join(video_output_directory, f"{video_name}_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        timer_count += fps / frames_captured_per_second # advances 52scenes
        cap.set(cv2.CAP_PROP_POS_FRAMES, timer_count)

        new_img = Image.open(frame_path)
        if not new_img.getbbox():
            os.remove(frame_path)
    print(f"Extracted {frame_count} frames to {video_output_directory}")
    return video_output_directory

