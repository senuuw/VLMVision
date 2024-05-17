import numpy as np
import pandas as pd
import pickle

def process_line(line, sample_voted, majority_vote_list):
    if line.startswith('Filename: '):
        current_filename = line.split(': ')[1].strip()
        sample_voted['File'].append(current_filename)
    elif line.startswith('['):
        # Clean line and just keep result1, result2, result3. Then split into list
        clean_line = line.strip().replace('[', '').replace(']', '').replace("'", '').split(", ")
        # Take majority vote
        majority_vote_list.append(max(set(clean_line), key=clean_line.count))

def categorize_samples(arrange_list, sample_voted):
    for sample in arrange_list:
        sample_voted['Setting'].append(sample[0] if sample[0] in ['outdoor', 'indoor'] else 'inconclusive')
        sample_voted['Lighting'].append(sample[1] if sample[1] in ['good', 'bad'] else 'inconclusive')
        sample_voted['Content Motion'].append(sample[2] if sample[2] in ['static', 'dynamic'] else 'inconclusive')

def create_dataframe(file_path):
    # Initialize the dictionary to hold the data
    sample_voted = {
        'File': [],
        'Setting': [],
        'Lighting': [],
        'Content Motion': []
    }

    majority_vote_list = []

    try:
        with open(file_path, "r") as file:
            for line in file:
                process_line(line, sample_voted, majority_vote_list)
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    arrange_list = np.array(majority_vote_list).reshape(-1, 3).tolist()
    categorize_samples(arrange_list, sample_voted)

    return pd.DataFrame(sample_voted)

sample_dataframe = create_dataframe('sampleresults.txt')