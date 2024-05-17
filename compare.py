import numpy as np
import pandas as pd
import pickle

def process_line(line, sample_voted, majority_vote_list):
    if line.startswith('Filename: '):
        current_filename = line.split(': ')[1].strip()
        sample_voted['File'].append(current_filename)
    elif line.startswith('['):
        clean_line = line.strip().replace('[', '').replace(']', '').replace("'", '').split(", ")
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


predictions = create_dataframe('sampleresults.txt')
ground_truth = pd.read_csv('sampletruth.csv')


class Metrics:
    def __init__(self, ground_truth, predictions):
        self.gt = ground_truth
        self.pred = predictions


    def summary(self):
        summary_data = {}
        total_conflicts = self.gt.compare(self.pred).shape[0]
        total_entries = self.gt.shape[0]
        percent_conflicts = round(total_conflicts / total_entries * 100, 2)
        summary_data['Conflicted Images'] = [total_conflicts, percent_conflicts]

        for column in self.gt.drop(columns=['File']).columns:
            conflict_count = self.gt[self.gt[column] != self.pred[column]].shape[0]
            percent_conflicts = round(conflict_count / total_entries * 100, 2)
            summary_data[column] = [conflict_count, percent_conflicts]

        summary_df = pd.DataFrame.from_dict(summary_data).T
        summary_df = summary_df.rename(columns={0: "Count", 1: "Percent"})
        print(summary_df.to_string())


    def display_conflict(self):
        compared_data = self.gt.compare(self.pred, keep_shape=True, keep_equal=False)
        compared_data['File'] = self.gt['File']

        conflict_dict = {'File': self.gt['File']}

        for column in self.gt.columns:
            if column != 'File':
                conflict_dict[f'gt_{column}'] = self.gt[column]
                conflict_dict[f'pred_{column}'] = self.pred[column]

        conflict_df = pd.DataFrame(conflict_dict)

        difference_mask = self.gt.drop(columns=['File']) != self.pred.drop(columns=['File'])
        rows_with_multiple_conflicts = difference_mask.sum(axis=1) >= 1
        conflict_df = conflict_df[rows_with_multiple_conflicts]

        print(conflict_df.to_string())




Results = Metrics(ground_truth, predictions)
Results.summary()
#print(ground_truth.compare(predictions))
#print(pd.merge(ground_truth, predictions, how='left', on=['File']))
