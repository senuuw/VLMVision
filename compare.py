import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load data and sort by 'Filename'
df = pd.read_pickle('results2.pkl').sort_values(by=['Filename'])
df_truth = pd.read_csv('frames_truth.csv').sort_values(by=['Filename'])

# Replace string boolean values with actual boolean values
df['Screens'] = df['Screens'].replace('False', False).replace('True', True)
df['People'] = df['People'].replace('False', False).replace('True', True)

# Define valid values for each category
valid_values = {
    'Setting': ['indoor', 'outdoor'],
    'Lighting': ['good', 'bad'],
    'People': [True, False],
    'Screens': [True, False]
}
categories = ['Setting', 'Lighting', 'People', 'Screens']

def df_compare(df, df_truth):
    # Initialize counts
    excluded_count = {category: 0 for category in valid_values}

    # Filter invalid predicted values and count them
    for category in valid_values:
        mask_pred = df[category].isin(valid_values[category])
        mask_true = df_truth[category].isin(valid_values[category])
        valid_mask = mask_pred & mask_true
        excluded_count[category] = (~valid_mask).sum()
        df = df[valid_mask]
        df_truth = df_truth[valid_mask]

    # Calculate confusion matrices for each category
    confusion_matrices = {}
    for category in categories:
        y_true = df_truth[category]
        if category == 'People' or category == 'Screens':
            y_pred = df[category].astype(bool)
        else:
            y_pred = df[category]
        cm = confusion_matrix(y_true, y_pred, labels=valid_values[category])
        confusion_matrices[category] = cm

    # Display confusion matrices and excluded counts
    for category, cm in confusion_matrices.items():
        labels = valid_values[category]
        cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in labels],
                             columns=[f'Predicted {i}' for i in labels])
        print(f"Confusion Matrix for {category}:")
        print(cm_df)
        print(f"Excluded Count for {category}: {excluded_count[category]}")
        print("\n")

def count_differences(df, df_truth):
    # Count differences
    compare = df[['Filename', 'Setting', 'Lighting', 'People', 'Screens']].compare(df_truth[['Filename', 'Setting', 'Lighting', 'People', 'Screens']])
    count = {category: 0 for category in categories}
    conflict_count = compare.shape[0]
    for category in categories:
        count[category] = conflict_count - sum(compare[(category, 'self')].isna())
    all_rows = df[['Filename', 'Setting', 'Lighting', 'People', 'Screens']].compare(
        df_truth[['Filename', 'Setting', 'Lighting', 'People', 'Screens']], keep_shape = True)
    row_count = all_rows.shape[0]
    print(f"All rows: {row_count}")
    print(f"Conflicts: {conflict_count}")
    print(f"Count: {count}")
# Compare original data
df_compare(df, df_truth)
count_differences(df, df_truth)

compare = df[['Filename', 'Setting', 'Lighting', 'People', 'Screens']].compare(df_truth[['Filename', 'Setting', 'Lighting', 'People', 'Screens']], keep_shape=True, keep_equal=True)
setting_diff = compare.loc[~(compare[('Setting', 'self')] == compare[('Setting', 'other')])]
people_diff = compare.loc[~(compare[('People', 'self')] == compare[('People', 'other')])]
lighting_diff = compare.loc[~(compare[('Lighting', 'self')] == compare[('Lighting', 'other')])]
screens_diff = compare.loc[~(compare[('Screens', 'self')] == compare[('Screens', 'other')])]

# Refine the dataframe
refinement_mask = (df['Blur'] < 1000) & (df['Bright Spot'] < 8000)
refined_df = df[refinement_mask]
refined_df_truth = df_truth[refinement_mask]

print("+++++++++++++++++++++++++++++++++++++++")
print('DATA FOR REMOVED BRIGHT SPOT AND BLUR')
# Compare refined data
df_compare(refined_df, refined_df_truth)
count_differences(refined_df, refined_df_truth)
pass