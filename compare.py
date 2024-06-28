import pandas as pd

df = pd.read_pickle('resultsclean.pkl')
df_sort = df.sort_values(by=['Filename'])

for name in list(df_sort['Filename']):
    print(name)
pass