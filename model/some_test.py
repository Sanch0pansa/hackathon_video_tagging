import pandas as pd

dataset = pd.read_csv('./train_dataset_tag_video/baseline/train_data_categories.csv')
dct = {}
for _, row in dataset.iterrows():
    tags = row['tags'] if str(row['tags']) != 'nan' else ''
    for tag in (tags.split(",")):
        tg = tag.strip().lower()
        dct[tg] = dct.get(tg, 0) + 1

data = [(key, dct[key]) for key in dct]
sm = sum([x[1] for x in data])
data = [(x[0], x[1] / sm) for x in data]

data.sort(key=lambda x: x[1], reverse=True)

print(*data[:10], sep="\n")