import pandas as pd
fraction=1
from sklearn.model_selection import train_test_split
train_data = pd.read_csv('./data/train.csv')
wiki_train_data,wiki_dev_data = train_test_split(train_data,test_size=0.2,random_state=42)
wiki_train_data = wiki_train_data.sample(frac=fraction)
wiki_dev_data = wiki_dev_data.sample(frac=fraction)
wiki_train_data.rename(columns={'toxic':'hurt','severe_toxic':'toxic','identity_hate':'hate'},inplace=True)
wiki_dev_data.rename(columns={'toxic':'hurt','severe_toxic':'toxic','identity_hate':'hate'},inplace=True)
wiki_dev_data.to_csv('./data/wikipedia_dev.csv',index=False)
wiki_train_data.to_csv('./data/wikipedia_train.csv',index=False)
test_data = pd.read_csv('./data/test.csv')
test_data_labels = pd.read_csv('./data/test_labels.csv')
test_data_labels = test_data_labels[test_data_labels['toxic']!=-1]
df = test_data.merge(test_data_labels, left_on='id', right_on='id',suffixes=(False, False))
df = df.sample(frac=fraction)
df.rename(columns={'toxic':'hurt','severe_toxic':'toxic','identity_hate':'hate'},inplace=True)
df.to_csv('./data/wikipedia_test.csv',index=False)