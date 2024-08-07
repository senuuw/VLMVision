import os
import pandas as pd
import numpy as np

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


def filter_optimal_scenes(data_dict, n):
    length = max([end for sublist in data_dict.values() for _, _, end in sublist]) + 1
    criteria = {
        'Setting': 'indoor',
        'Lighting': 'good',
        'People': False,
        'Screens': False
    }

    valid_indices = set(range(length))

    for key, value in criteria.items():
        indices = set()
        for val, start, end in data_dict[key]:
            if val == value:
                indices.update(range(start, end + 1))
        valid_indices &= indices

    valid_indices = sorted(valid_indices)

    # Find consecutive sequences of at least length n
    result = []
    temp_sequence = []

    for i in range(len(valid_indices)):
        if not temp_sequence or valid_indices[i] == temp_sequence[-1] + 1:
            temp_sequence.append(valid_indices[i])
        else:
            if len(temp_sequence) >= n:
                result.append(temp_sequence)
            temp_sequence = [valid_indices[i]]

    if len(temp_sequence) >= n:
        result.append(temp_sequence)

    return result

def show_filtered_blocks(directory_path, window_size , min_length):
    directory_path = '/home/sebastian/VLMVision/ego4d/results'
    indoor_results_paths = os.listdir(directory_path)
    for path in indoor_results_paths:
        result_path = os.path.join(directory_path, path)
        segment_dict = classify_segments(result_path, window_size)
        segment_blocks = create_segment_blocks(segment_dict)
        filtered = filter_optimal_scenes(segment_blocks, 1)
        print(path)
        print(filtered)

directory_path = '/home/sebastian/VLMVision/ego4d/results'

show_filtered_blocks(directory_path, 5, 1)