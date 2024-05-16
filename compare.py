import numpy as np
import pandas as pd

def create_dataframe(file_path):
    # This is terrible and a result of my poor planning. Hopefully I only have to do this once

    sample_voted = {
        'File': [],
        'Setting': [],
        'Lighting': [],
        'Content Motion': []
    }

    file = open(file_path, "r")
    majority_vote_list = []

    for line in file:
        if line.startswith('Filename: '):
            current_filename = line.split(': ')[1].strip()
            sample_voted['File'].append(current_filename)
        elif line.startswith('['):
            # Clean line and just keep result1, result2, result3. Then split into list
            clean_line = line.strip().replace('[', '').replace(']', '').replace("'", '').split(", ")
            # Take majority vote
            majority_vote_list.append((max(set(clean_line), key=clean_line.count)))
        else:
            pass

    arrange_list = np.array(majority_vote_list).reshape(-1,3).tolist()
    for sample in arrange_list:
        if sample[0] in ['outdoor', 'indoor']:
            sample_voted['Setting'].append(sample[0])
        else:
            sample_voted['Setting'].append('inconclusive')

        if sample[1] in ['good', 'bad']:
            sample_voted['Lighting'].append(sample[1])
        else:
            sample_voted['Lighting'].append('inconclusive')

        if sample[2] in ['static', 'dynamic']:
            sample_voted['Content Motion'].append(sample[2])
        else:
            sample_voted['Content Motion'].append('inconclusive')

    return pd.DataFrame(sample_voted)

sample_dataframe = create_dataframe('sampleresults.txt')
print(sample_dataframe.to_string())
