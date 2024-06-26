import cv2
import os




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


def extract_directory_frames(path, n):
    video_count = 0
    videos = os.listdir(path)[1:n+1] #start at 1 to ignore manifest)
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
                        cv2.imwrite(f'frames/{video[:-4]}_{frame_count:04d}.jpg', frame)
                        timer_count += 600 # i.e. at 30 fps, this advances one second, currently every 20s
                        cap.set(cv2.CAP_PROP_POS_FRAMES, timer_count)
                else:
                    cap.release()
                    break
            except:
                print(f'Video {video_count}, path:{current_vid_path} failed cap.read()')

extract_directory_frames(r"D:\ego4d\v2\full_scale", 30)