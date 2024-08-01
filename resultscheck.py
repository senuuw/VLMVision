import os

import pandas as pd

directory_path = r'C:\Users\Sebastian\Desktop\researchvideos\results'
file_name_list = os.listdir(directory_path)
def analyze_pickles(directory_path):
    good_list = []
    pd.set_option('future.no_silent_downcasting', True)
    for file_name in file_name_list:
        pickle_path = os.path.join(directory_path, file_name)
        df = pd.read_pickle(pickle_path)
        df['Screens'] = df['Screens'].replace('False', False).replace('True', True)
        df['People'] = df['People'].replace('False', False).replace('True', True)
        total_rows = df.shape[0]
        indoor_prop = df[df['Setting'] == 'indoor'].shape[0] /  total_rows
        good_lighting_prop = df[df['Lighting'] == 'good'].shape[0] /  total_rows
        no_people_prop = df[df['People'] == False].shape[0] / total_rows
        if indoor_prop >= 0.80 and good_lighting_prop >= 0.80:
            print(f"{file_name}, indoor:{indoor_prop:.2f}, lighting:{good_lighting_prop:.2f}")
    return good_list[]

analyze_pickles(directory_path)