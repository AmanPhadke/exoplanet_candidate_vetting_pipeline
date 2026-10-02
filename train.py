import pandas as pd
import numpy as np
import pickle

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Loading Parameters
xgb_params = {
        'eta' : 0.1,
        'max_depth' : 10,
        'min_child_weight' : 5, 

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1
    }

output = f'xgb_model.bin'


# Data Preparation


print('Preparing Data....')
data = "C:/Users/PC/OneDrive/Desktop/Exoplanet_Vetting/exoplanet_candidate_vetting_pipeline/dataset.csv"
df = pd.read_csv(data)

features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad', 'tfopwg_disp']

df = df[features]

df['tfopwg_disp'].unique()

df = df[df['tfopwg_disp'].isin(['FP', 'KP', 'FA', 'CP'])].copy()
df = df.reset_index(drop=True)
df['label'] = df['tfopwg_disp'].isin(['KP', 'CP']).astype('int') 

df.label.value_counts()
del df['tfopwg_disp']

columns = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']

for col in columns:
    df[col] = df[col].fillna(df[col].median())


print('Creating Train Test Split....')
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state = 1)


# Training the model
print('Training the model....')

def train(df_full_train, y_train, xgb_params):
    features_xgb = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
                'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']

    dtrain = xgb.DMatrix(df_full_train[features_xgb], label = y_train, feature_names = features_xgb)

    xgb_model = xgb.train(xgb_params,
                          dtrain, 
                          num_boost_round = 200,
                          verbose_eval = 10)

    return xgb_model, features_xgb


def predict(df_test, model, features_xgb):
    dtest = xgb.DMatrix(df_test[features_xgb], feature_names = features_xgb)

    return model.predict(dtest)


print('Training the final model......')
model, features_xgb = train(df_full_train, df_full_train.label.values, xgb_params)
y_pred = predict(df_test, model, features_xgb)


print('Calculating AUC Score......')
y_test = df_test.label.values
auc = roc_auc_score(y_test, y_pred)

print(f'AUC Score: {auc}')


print('Saving the model......')
with open(output, 'wb') as f_out:
    pickle.dump((model, features_xgb), f_out)