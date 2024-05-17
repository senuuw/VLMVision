import cv2
import os

videos = os.listdir(r"D:\ego4d\v2\full_scale")[10:20]
print(videos)



def extract_frames(path, video_num):
    count = 0
    frame_count = 0
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            #try to open video, if error then skip video
            if ret:
                if frame_count == 5:
                    break
                else:
                    frame_count += 1
                    cv2.imwrite(f'sampleframes/video{video_num}_frame{frame_count}.jpg', frame)
                    count += 600 # i.e. at 30 fps, this advances one second, currently every 20s
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            else:
                cap.release()
                break
        except:
            print(f'Video {video_num}, path:{path} failed cap.read()')


def extract_indexvideo_frames(start_n, end_n, path):
    video_count = start_n
    videos = os.listdir(path)[start_n:end_n]
    for video in videos:
        count = 0
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
                        cv2.imwrite(f'sampleframes/video{video_count}_frame{frame_count}.jpg', frame)
                        count += 600 # i.e. at 30 fps, this advances one second, currently every 20s
                        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                else:
                    cap.release()
                    break
            except:
                print(f'Video {video_count}, path:{current_vid_path} failed cap.read()')

extract_frames(r"D:\ego4d\v2\full_scale\29a90a49-4137-4f81-b24d-2d923ab8fb55.mp4", 21)