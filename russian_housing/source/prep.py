import pandas as pd


raw_dir = '../data/raw/'
processed_dir = '../data/processed/'
df_train = pd.read_csv(raw_dir + 'train.csv')
df_test = pd.read_csv(raw_dir + 'test.csv')
df_macro = pd.read_csv(raw_dir + 'macro.csv') 


def merge_macro_df(df):
	return pd.merge(df, df_macro, on=['timestamp'])

df_macro_train =  merge_macro_df(df_train)
df_macro_test = merge_macro_df(df_test)

df_macro_train.to_pickle(processed_dir + 'macro_train.pkl')
df_macro_test.to_pickle(processed_dir + 'macro_test.pkl')

