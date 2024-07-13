import os

import pandas as pd
import numpy as np

df = pd.read_pickle('test_set_small/testvids_results_small/0a81d795-8261-4059-8afa-d302084b1aab_clip.pkl')
pd.set_option('future.no_silent_downcasting', True)
df['Screens'] = df['Screens'].replace('False', False).replace('True', True)
df['People'] = df['People'].replace('False', False).replace('True', True)
list(df['Setting'])

valid_values = {
    'Setting': ['indoor', 'outdoor'],
    'Lighting': ['good', 'bad'],
    'People': [True, False],
    'Screens': [True, False]
}

category_list = ['Setting', 'Lighting', 'People']

def most_common_response(query_list, category, n):
    frequency_dict = {
        'Inconclusive': 0
    }
    for value in valid_values[category]:
        frequency_dict[value] = 0

    for item in query_list:
        if item in valid_values[category]:
            frequency_dict[item] += 1
        else:
            frequency_dict['Inconclusive'] += 1\

    most_frequent = max(frequency_dict, key=frequency_dict.get)
    if frequency_dict[most_frequent] >= np.ceil(n - np.sqrt(n)):
        return most_frequent
    else:
        return 'inconclusive'



def classify_segments(df_path, n):
    df = pd.read_pickle(df_path)
    pd.set_option('future.no_silent_downcasting', True)
    df['Screens'] = df['Screens'].replace('False', False).replace('True', True)
    df['People'] = df['People'].replace('False', False).replace('True', True)
    segment_dict = {
        'Setting': [],
        'Lighting': [],
        'People': [],
        'Screens': [],
    }
    for category in list(segment_dict.keys()):
        time = 0
        classify_list = list(df[category])
        sublists = [classify_list[i:i + n] for i in range(len(classify_list) - n + 1)]
        for sublist in sublists:
            segment_dict[category].append((most_common_response(sublist, category, n), time, time+n))
            time += 1
    return segment_dict


def create_segment_blocks_overlap(segment_dict):
    segment_block_dict = {}

    for category in segment_dict.keys():
        current_segment_list = segment_dict[category]
        segment_block_dict[category] = []

        # Initialize the first segment
        segment_response = current_segment_list[0][0]
        segment_start_time = current_segment_list[0][1]
        segment_end_time = current_segment_list[0][2]

        for i in range(len(current_segment_list) - 1):
            if segment_response != current_segment_list[i + 1][0]:
                segment_block_dict[category].append([segment_response, segment_start_time, segment_end_time])
                segment_response = current_segment_list[i + 1][0]
                segment_start_time = current_segment_list[i + 1][1]
                segment_end_time = current_segment_list[i + 1][2]
            else:
                segment_end_time = current_segment_list[i + 1][2]

        # Append the last segment
        segment_block_dict[category].append([segment_response, segment_start_time, segment_end_time])

    return segment_block_dict

def create_segment_blocks(segment_dict):
    segment_block_dict = {}

    for category in segment_dict.keys():
        current_segment_list = segment_dict[category]
        segment_block_dict[category] = []

        # Initialize the first segment
        segment_response = current_segment_list[0][0]
        segment_start_time = current_segment_list[0][1]
        segment_end_time = current_segment_list[0][2]

        for i in range(len(current_segment_list) - 1):
            next_response = current_segment_list[i + 1][0]
            next_start_time = current_segment_list[i + 1][1]
            next_end_time = current_segment_list[i + 1][2]

            if segment_response != next_response:
                if segment_end_time >= next_start_time:
                    # Split the overlap
                    mid_time = (segment_end_time + next_start_time) // 2
                    segment_block_dict[category].append([segment_response, segment_start_time, mid_time])
                    segment_response = next_response
                    segment_start_time = mid_time + 1
                    segment_end_time = next_end_time
                else:
                    segment_block_dict[category].append([segment_response, segment_start_time, segment_end_time])
                    segment_response = next_response
                    segment_start_time = next_start_time
                    segment_end_time = next_end_time
            else:
                segment_end_time = next_end_time

        # Append the last segment
        segment_block_dict[category].append([segment_response, segment_start_time, segment_end_time])

    return segment_block_dict

def segment_block_folder(pickle_directory):
    file_list = os.listdir(pickle_directory)
    for pickle_file_name in file_list:
        current_pickle_path = os.path.join(pickle_directory, pickle_file_name)
        segment_dict = classify_segments(current_pickle_path, 5)
        print(pickle_file_name)
        for category in category_list:
            print(create_segment_blocks(segment_dict)[category])

#segment_block_folder('testvids_results')