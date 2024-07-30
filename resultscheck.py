import pandas as pd

path = '/home/sebastian/VLMVision/ego4d/results/3e08beb0-9108-4e77-b2ae-80f91ceac474.pkl'
df = pd.read_pickle(path)
print(df)