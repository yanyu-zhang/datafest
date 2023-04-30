import pandas as pd
import os

# split the questionposts.csv into 10 files

ROW_PER_FILE = 50000

if os.path.exists('./tmp'):
    os.system('rm -rf ./tmp')
os.mkdir('./tmp')

for i in range(10):
    df = pd.read_csv('./data/questionposts.csv', skiprows=range(1, i * ROW_PER_FILE), nrows=ROW_PER_FILE)
    if df.count().max() == 0:
        break
    df = df.iloc[:, :5]
    df.to_csv('./tmp/questionposts' + str(i) + '.csv', index=False)

