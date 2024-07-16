from classifyvideo import classify_segments, create_segment_blocks, create_segment_blocks_overlap
import cv2
import pandas as pd

def annotate_video_with_classifications(video_path, segment_dict, output_path):
    segment_block_dict = create_segment_blocks_overlap(segment_dict)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current time in seconds
        current_time = frame_number / fps

        annotations = []
        for category, segments in segment_block_dict.items():
            for segment in segments:
                segment_response, start_time, end_time = segment
                if start_time <= current_time <= end_time:
                    annotations.append(f"{category}: {segment_response}")
                    break

        # Annotate frame with the current classifications
        fontScale = 1  # Font scale for text
        thickness = 2  # Thickness of text
        y_offset = 50  # Starting Y position for text
        for annotation in annotations:
            text_x = 10  # X-coordinate for the top-left corner
            text_y = y_offset  # Y-coordinate for the top-left corner
            # Draw the text with a black border
            cv2.putText(frame, annotation, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            # Draw the white text on top of the black border
            cv2.putText(frame, annotation, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
            y_offset += 30  # Move to the next line for the next annotation

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to {output_path}")


def annotate_video_with_classifications_dataframe(video_path, pickle_path, output_path, fps=30):
    # Load the pickle file into a DataFrame
    df = pd.read_pickle(pickle_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    classification_interval = fps  # 30 52scenes per second

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Determine the corresponding row in the DataFrame
        row_number = frame_number // classification_interval
        if row_number < len(df):
            annotations = df.iloc[row_number]

            # Annotate frame with the current classifications
            fontScale = 1  # Font scale for text
            thickness = 2  # Thickness of text
            y_offset = 50  # Starting Y position for text
            for column in df.columns[1:]:  # Skip the Filename column
                if column not in ['Blur', 'Bright Spot']:  # Skip blur and bright spot columns
                    annotation = f"{column}: {annotations[column]}"
                    text_x = 10  # X-coordinate for the top-left corner
                    text_y = y_offset  # Y-coordinate for the top-left corner
                    # Draw the text with a black border
                    cv2.putText(frame, annotation, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    # Draw the white text on top of the black border
                    cv2.putText(frame, annotation, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
                    y_offset += 30  # Move to the next line for the next annotation

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to {output_path}")


name = '0a81d795-8261-4059-8afa-d302084b1aab'

for name in ['0a81d795-8261-4059-8afa-d302084b1aab', '0c0a685f-38d7-4dfe-9404-fe508894eb2d', '1cdc92fa-50cd-4461-adf2-ece8cb2a5d31',
             '2af7c658-0b4e-4f3c-b72e-ab77a06e1765', '4b0025fd-2d90-44ca-850b-e0d2131f6af1']:
    video_path = f'testvids/{name}_clip.mp4'
    dataframe_path = f'testvids_results/{name}_clip.pkl'
    output_path_raw = f'annotatedvideosdataframe/{name}_clip_annotated.avi'
    output_path = f'annotatedvideos/{name}_clip_annotated.avi'
    annotate_video_with_classifications_dataframe(video_path, dataframe_path, output_path_raw)
    segment_dict = classify_segments(dataframe_path,5)
    annotate_video_with_classifications(video_path, segment_dict, output_path)
    print(f'done with {name}')