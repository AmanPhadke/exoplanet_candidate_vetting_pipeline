#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[2]:


data = "D:/Space Proj/dataset.csv"


# In[3]:


df = pd.read_csv(data)


# ! [image.png](attachment:5ee474fb-e936-46c3-a681-28da1ebc9755.png) 

# In[4]:


df


# ### Dropping some non-useful features from the datset such as
# - tid, toi: Just numbers and no predictive significance
# - toi_created, rowupdate: record dates
# - pl_tranmid: the timestamp of a specific transit event, not a reusable property
# - All the other unncessary err columns: they're uncertainty companions, not new signals so including them roughly triples your feature count

# ### Dropping another set of features from the dataset on purpose
# - ra, rastr, dec, decstr: these are sky coordinates and including these risks the model learning "where known planets cluster in the sky" rather than the actual transit physics. Which could help the model "cheat" during predictions
# - st_pmra, st_pmdec (and their uncertainty columns): proper motion meaning how fast the star drifts across the sky over years. Not physically related to whether a transit is real.

# # Final Feature List
# ## Transit Signal: 
# - pl_orbper(Orbital Period) : How often does the dip appear. As the planet orbits the star just like earth orbits the sun. An orbital period is a consistent "year" it orbits it's star. For a real planet the orbital period is consistent for every transit.
# 
# - pl_trandurh (Transit Duration) : How long each individual dip lasts from the moment planet starts blocking light to when it fully clears
# 
# - pl_trandep (Transit Depth) : How much the brightness drop. This is really important feature as bigger planets block more light, so this is the main clue for size
# 
# - pl_rade (Planet Radius) : The planet's physical size, calculated from the transit depth combined with how big we know the host star is (derived value using the transit depth).
# 
# - pl_insol (Insolution) : So it's basically much solar radiation the planet receives, relative to what Earth gets from the Sun
# 
# - pl_eqt (Equillibrium Temperature) : The planet's estimated surface temperature if it had no atmosphere at all (calculated from insolation as higher the isolation higher the planet's temperature)
# 
# ## Star:
# - st_tmag (TESS Magnitude) - How bright the star appears to us, on TESS's brightness scale
# - st_dist (Stellar Distance) - How far away the star is from Earth.
# - st_teff (Effective Temperature) - The star's surface temperature. This tells you what kind of star it is (cold star = red/orange colored, hot star = blue/white color, our sun is a cold star that's why it is orange in color)
# - st_logg (Surface Gravity) : A log-scale measure of how strong gravity is at the star's surface
# - st_rad (Stellar Radius) : The star's physical size, relative to our Sun

# In[5]:


features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad', 'tfopwg_disp']


# In[6]:


df = df[features]


# In[7]:


df['tfopwg_disp'].unique()


# In[10]:


df = df[df['tfopwg_disp'].isin(['FP', 'KP', 'FA', 'CP'])].copy()
df = df.reset_index(drop=True)
df['label'] = df['tfopwg_disp'].isin(['KP', 'CP']).astype('int') #We are encoding the 'KP' and 'CP' as Planets (1) and 'FA' and 'FP' as Not Planets (0)


# In[11]:


df.label.value_counts() # We have a fairly balanced dataset


# In[12]:


del df['tfopwg_disp']


# In[13]:


df.dtypes  # As all the datatypes are numerical we can easily handle the missing values


# In[14]:


columns = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']

for col in columns:
    df[col] = df[col].fillna(df[col].median())


# In[15]:


df.describe()


# In[16]:


# As we can see that there are no sentinel values


# In[17]:


df.isna().sum()


# # EDA

# ## Correlation

# In[18]:


corr_matrix = df.corr()


# In[19]:


corr_matrix


# In[20]:


mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper_half = corr_matrix.where(mask)


# In[21]:


unique_pairs = upper_half.unstack().dropna()


# In[22]:


unique_pairs.sort_values(ascending=False).head(10)


# ### Here we can see some obvious observations
# 1. The Stellar Radius is highly correlated to the Planet Radius because planet radius is derived from star's radius
# 2. The host star's effective temperature (st_teff) has a positive correlation with the insolution (pl_insol) which we know as its is derived from the insolution. Eventually giving us high correlation between st_teff (star's effective temp) and pl_eqt (planet's equillibrium temp) because the higher the star's temperature the higher the planet's eq. temperature
# 3. st_tmag(Star's Brightness) and pl_trandep (planet's transit depth) also has a moderate correlation
# 4. st_logg (star's surface gravity) carries real predictive signal for finding the correct label

# In[23]:


features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']


# In[24]:


corr_label = df[features].corrwith(df.label)


# In[25]:


corr_label.sort_values(ascending=False)


# As we can see that Star's surface gravity, planet's transit duration and depth and orbital period are possitively correlated in the classification 
# whereas Planet's temperature and star's stellar distance has a moderate negative correlation

# # Logistic Regression

# In[26]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state = 1)


# In[27]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)


# In[28]:


df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[29]:


y_train = df_train['label'].values
y_val = df_val['label'].values
y_test = df_test['label'].values


# In[30]:


del df_train['label']
del df_val['label']
del df_test['label']


# In[31]:


# As we have all the numerical features there is no need for encoding
# Although we still need to rescale the values as each feature have major outliers which can disrupt the models predictions
from sklearn.preprocessing import StandardScaler


# In[32]:


model = LogisticRegression(class_weight='balanced')


# In[33]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train)
X_val_scaled = scaler.transform(df_val)


# In[34]:


model.fit(X_train_scaled, y_train)


# In[35]:


lr_preds = model.predict_proba(X_val_scaled)[:, 1]


# In[36]:


# Validating auc score for each possible threshold
thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    prediction = (lr_preds >= t)
    score = roc_auc_score(y_val, prediction)
    scores.append((t, score))


df__lr_scores = pd.DataFrame(scores, columns = ['threshold', 'score'])


# In[37]:


df__lr_scores.sort_values(by='score', ascending=False).head(3)


# In[38]:


model.score(X_val_scaled, y_val)


# In[39]:


def train(df_train, y_train, C=1.0):
    X_train_scaled = scaler.fit_transform(df_train)
    model = LogisticRegression(C=C, max_iter = 1000)
    model.fit(X_train_scaled, y_train)
    return model


# In[40]:


def predict(df_val, model):
    X_val_scaled = scaler.transform(df_val)
    y_preds = model.predict_proba(X_val_scaled)[:,1]
    return y_preds


# In[41]:


scores = []
for C in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]:
    model = train(df_train, y_train, C)
    y_pred = predict(df_val, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append((C, auc))


# In[42]:


scores # Scores plateau after C=1 and barely improves so C=5 would be a better pick for the model


# ### Final Logistic Regression Model:

# In[43]:


def logistic_regression(df_train, y_train, C=5.0, threshold = 0.50):
    X_train_scaled = scaler.fit_transform(df_train)
    model = LogisticRegression(class_weight = 'balanced', 
                               C=C, 
                               max_iter = 1000)

    model.fit(X_train_scaled, y_train)
    X_val_scaled = scaler.transform(df_val)
    y_preds = model.predict_proba(X_val_scaled)[:,1]
    prediction = (y_preds >= threshold)

    return prediction


# In[44]:


prediction = logistic_regression(df_train, y_train)
roc_auc_score(y_val, prediction)


# ### Confusion Matrix

# In[45]:


actual_pos = (y_val == 1)
actual_neg = (y_val == 0)
predict_pos = (prediction == 1)
predict_neg = (prediction == 0)


# In[46]:


tp = (predict_pos & actual_pos).sum()
tn = (predict_neg & actual_neg).sum() 
fp = (predict_pos & actual_neg).sum() 
fn = (predict_neg & actual_pos).sum() 


# In[47]:


confusion_matrix = np.array(
    [[tn, fp],
    [fn, tp]]
)


# In[48]:


confusion_matrix


# In[49]:


precision = tp / (tp + fp)
recall = tp / (tp + fn)


# In[50]:


print(f'Precision: {precision.round(3)* 100}%')
print(f'Recall: {recall.round(4)*100}%')


# # Decision Tree

# In[51]:


dt = DecisionTreeClassifier()


# In[52]:


dt.fit(X_train_scaled, y_train)


# In[53]:


dt_preds = dt.predict_proba(X_val_scaled)[:, 1]


# In[54]:


roc_auc_score(y_val, dt_preds)


# ### We are already Getting better accuracy than untuned Logistic Regression with decision tree

# ## Tuning decision tree parameters

# ### Depth Tuning

# In[55]:


scores = []

for max_depth in [1,2,3,4,5,6,10,15,50,100, None]:
    dt = DecisionTreeClassifier(max_depth = max_depth)
    dt.fit(X_train_scaled, y_train)
    dt_preds = dt.predict_proba(X_val_scaled)[:, 1]
    score = roc_auc_score(y_val, dt_preds)
    scores.append((score, max_depth))


# In[56]:


df_scores = pd.DataFrame(scores, columns=['auc', 'max_depth'])


# In[57]:


df_scores


# In[58]:


from sklearn.tree import plot_tree


# In[59]:


dt = DecisionTreeClassifier(max_depth = 5)
tree = dt.fit(X_train_scaled, y_train)


# ### Min Samples Leaf Tuning

# In[60]:


scores = []

for min_samples_leaf in [1,2,3,4,5,10, 15, 50, 100]:
    for max_depth in [1,2,3,4,5,6,10,15,20,None]:
        dt = DecisionTreeClassifier(max_depth = max_depth,
                                   min_samples_leaf = min_samples_leaf)
        dt.fit(X_train_scaled, y_train)
        dt_preds = dt.predict_proba(X_val_scaled)[:, 1]
        score = roc_auc_score(y_val, dt_preds)
        scores.append((round(score,3), max_depth, min_samples_leaf))


# In[61]:


df_scores = pd.DataFrame(scores, columns=['auc', 'max_depth', 'min_samples_leaf']) 


# In[62]:


df_scores.sort_values(by='auc', ascending=False).head(10)


# In[63]:


df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns= ['max_depth'], values='auc')
df_scores_pivot


# In[64]:


sns.heatmap(df_scores_pivot, annot=True, fmt='.2f')


# ### We'll be choosing max depth 6 and min_sample leaf 50

# ## Final Decision Tree Model

# In[65]:


def decision_tree(df_train, df_val, y_train, y_val, max_depth=6.0, min_samples_leaf= 50):
    X_train_scaled = scaler.fit_transform(df_train) 
    dt = DecisionTreeClassifier(max_depth = max_depth,
                                   min_samples_leaf = min_samples_leaf)

    X_val_scaled = scaler.transform(df_val)
    dt.fit(X_train_scaled, y_train)
    dt_preds = dt.predict_proba(X_val_scaled)[:, 1]

    return dt_preds


# # Random Forest

# In[66]:


rf = RandomForestClassifier(n_estimators = 10, random_state=42)


# In[67]:


rf.fit(X_train_scaled, y_train)


# In[68]:


rf_preds = rf.predict_proba(X_val_scaled)[:, 1]


# In[69]:


roc_auc_score(y_val, rf_preds)


# In[205]:


scores = []

for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators = n, random_state=42) 
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val , rf_pred)
    scores.append((n, auc))


# In[206]:


df_scores = pd.DataFrame(scores, columns=['n_estimators','auc'])


# In[207]:


plt.plot(df_scores.n_estimators, df_scores.auc)
plt.savefig("rf_estimator_graph.png", dpi=300, bbox_inches="tight")
plt.show()


# In[208]:


scores = []
for n in range(10, 201, 10):
    for d in [1, 5, 10 ,15, 20, 50, 100]:
        rf = RandomForestClassifier(n_estimators = n,
                                    max_depth = d,
                                    random_state=42) 
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict_proba(X_val_scaled)[:,1]
        auc = roc_auc_score(y_val , rf_pred)
        scores.append((n, d, auc))


# In[209]:


df_scores = pd.DataFrame(scores, columns=['n_estimators', 'max_depth','auc'])


# In[210]:


df_scores.sort_values(by='auc', ascending=False).head(5)


# In[212]:


for d in [10 ,15, 20, 50]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.auc, label= f'max_depth ={d}')

plt.legend()
plt.savefig("rf_max_depth.png", dpi=300, bbox_inches="tight")
plt.show()


# In[213]:


scores = []
for n in range(10, 201, 10):
    for s in [1, 5, 10 ,15, 20, 50, 100]:
        rf = RandomForestClassifier(n_estimators = n,
                                    max_depth = 15,
                                    min_samples_leaf = s,
                                    random_state=42) 
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict_proba(X_val_scaled)[:,1]
        auc = roc_auc_score(y_val , rf_pred)
        scores.append((n, s, auc))


# In[214]:


df_scores = pd.DataFrame(scores, columns=['n_estimators', 'min_samples_leaf','auc'])


# In[215]:


for s in [1, 5, 10 ,15, 20, 50, 100]:
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    plt.plot(df_subset.n_estimators, df_subset.auc, label= f'min_samples_leaf ={s}')

plt.legend()
plt.savefig("rf_msl.png", dpi=300, bbox_inches="tight")
plt.show()


# In[80]:


df


# In[81]:


temp_df = pd.read_csv(data)


# In[82]:


features = ['tid', 'pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad', 'tfopwg_disp']


# In[83]:


temp_df = temp_df[features]


# In[84]:


m = temp_df[temp_df['tfopwg_disp'].isin(['FP', 'KP', 'FA', 'CP'])]

print(m['tid'].duplicated().sum())          # rows whose star already appeared
print(m['tid'].nunique(), len(m))           # unique stars vs total rows
print(m['tid'].value_counts().head(10))     # the worst offenders


# In[85]:


from sklearn.model_selection import GroupKFold, cross_val_score

groups = m['tid'].reset_index(drop=True)
X_new = df.drop(columns='label')
y_new = df['label']
X_scaled_new = scaler.fit_transform(X_new)


rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, random_state=1, n_jobs=-1)
scores = cross_val_score(rf, X_scaled_new, y_new, groups=groups, cv=GroupKFold(n_splits=5), scoring='roc_auc')
print(scores.mean(), scores.std())


# # XGBoost

# In[86]:


get_ipython().system('pip install xgboost')


# In[87]:


import xgboost as xgb


# In[88]:


features_xgb = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']


# In[89]:


X_train_scaled = scaler.fit_transform(df_train)
X_val_scaled = scaler.transform(df_val)


# In[90]:


dtrain = xgb.DMatrix(X_train_scaled, label = y_train, feature_names = features_xgb)
dval = xgb.DMatrix(X_val_scaled, feature_names = features_xgb)


# In[91]:


watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[92]:


def parse_output_xgb(output):
    tree = []
    trains = []
    vals = []
    scores = []
    for line in output.stdout.strip().split('\n'):
        num_iter, train_auc, val_auc = line.split('\t')
        it = int(num_iter.strip('[]'))
        train = float(train_auc.strip().split(':')[1])
        val = float(val_auc.strip().split(':')[1])

        tree.append(it)
        trains.append(train)
        vals.append(val)
        scores.append((it, train, val))


    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(scores, columns = columns)

    return df_results


# In[103]:


scores_xgb_eta = {}


# In[94]:


scores_xgb_max_depth = {}


# In[95]:


scores_xgb_min_child_weight = {}


# In[104]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta' : 0.1,\n    'max_depth' : 10,\n    'min_child_weight' : 5, \n\n    'objective': 'binary:logistic',\n    'nthread': 8,\n    'eval_metric': 'auc',\n\n    'seed': 1,\n    'verbosity': 1\n}\n\nxgb_model = xgb.train(xgb_params,\n                      dtrain, \n                      num_boost_round = 200,\n                      verbose_eval = 10,\n                      evals=watchlist)\n")


# In[97]:


xgb_preds = xgb_model.predict(dval)


# In[98]:


key = f'min_child_weight={xgb_params['min_child_weight']}'
scores_xgb_min_child_weight[key] = parse_output_xgb(output)


# In[99]:


roc_auc_score(y_val, xgb_preds)


# In[100]:


from IPython.utils import capture


# In[102]:


for key, df_score in scores_xgb_eta.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label = key)

plt.legend()


# In[ ]:


for key, df_score in scores_xgb_max_depth.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label = key)

plt.ylim(0.82, 0.93)
plt.legend()


# In[ ]:


for key, df_score in scores_xgb_min_child_weight.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label = key)

plt.ylim(0.82, 0.93)
plt.legend()


# ## Final XGBoost Model

# In[105]:


xgb_params = {
    'eta' : 0.1,
    'max_depth' : 10,
    'min_child_weight' : 5, 

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1
}

xgb_model = xgb.train(xgb_params,
                      dtrain, 
                      num_boost_round = 200,
                      verbose_eval = 10)


# # Final Evaluation of all the models

# In[173]:


def logistic_regression(df_train, y_train, df_test, C=5.0, threshold = 0.50):
    X_train_scaled = scaler.fit_transform(df_train)
    lr_model = LogisticRegression(class_weight = 'balanced', 
                               C=C, 
                               max_iter = 2000)

    lr_model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(df_test)
    y_preds = lr_model.predict_proba(X_test_scaled)[:,1]

    return y_preds, lr_model


# In[165]:


def decision_tree(df_train, y_train, df_test, max_depth=6, min_samples_leaf= 50):
    dt = DecisionTreeClassifier(max_depth = max_depth,
                                min_samples_leaf = min_samples_leaf)

    dt.fit(df_train, y_train)

    dt_preds = dt.predict_proba(df_test)[:, 1]

    return dt_preds, dt


# In[168]:


def random_forest(df_train, y_train, df_test):
    rf = RandomForestClassifier(n_estimators = 150,
                                        max_depth = 15,
                                        min_samples_leaf = 5,
                                        random_state=42) 
    rf.fit(df_train, y_train)
    rf_pred = rf.predict_proba(df_test)[:,1]
    return rf_pred, rf


# In[167]:


def xgboost(df_train, y_train, df_test):
    features_xgb = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
                'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']

    dtrain = xgb.DMatrix(df_train, label = y_train, feature_names = features_xgb)
    dtest = xgb.DMatrix(df_test, feature_names = features_xgb)

    xgb_params = {
        'eta' : 0.1,
        'max_depth' : 10,
        'min_child_weight' : 5, 

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1
    }

    xgb_model = xgb.train(xgb_params,
                          dtrain, 
                          num_boost_round = 200,
                          verbose_eval = 10)

    xgb_preds = xgb_model.predict(dtest)
    return xgb_preds, xgb_model


# In[110]:


y_full_train = df_full_train['label'].values


# In[111]:


del df_full_train['label']


# In[148]:


lr_preds, lr_model = logistic_regression(df_full_train, y_full_train, df_test)
dt_preds, dt_model = decision_tree(df_full_train, y_full_train, df_test)
rf_preds, rf_model = random_forest(df_full_train, y_full_train, df_test)
xgb_preds, xgb_model = xgboost(df_full_train, y_full_train, df_test)


# In[149]:


print("Logistic Regression:", round(roc_auc_score(y_test, lr_preds),3))
print("Decision Tree:", round(roc_auc_score(y_test, dt_preds), 3))
print("Random Forest:", round(roc_auc_score(y_test, rf_preds),3))
print("XGBoost:", round(roc_auc_score(y_test, xgb_preds),3))


# In[174]:


lr_preds_val, lr_model = logistic_regression(df_train, y_train, df_val)
dt_preds_val, dt_model = decision_tree(df_train, y_train, df_val)
rf_preds_val, rf_model = random_forest(df_train, y_train, df_val)
xgb_preds_val, xgb_model = xgboost(df_train, y_train, df_val)


# In[ ]:


print("Logistic Regression:", round(roc_auc_score(y_val, lr_preds_val),3))
print("Decision Tree:", round(roc_auc_score(y_val, dt_preds_val), 3))
print("Random Forest:", round(roc_auc_score(y_val, rf_preds_val),3))
print("XGBoost:", round(roc_auc_score(y_val, xgb_preds_val),3))


# # Group K Cross Validation

# In[202]:


temp_df


# In[120]:


temp_df = temp_df[temp_df['tfopwg_disp'].isin(['FP', 'KP', 'FA', 'CP'])].copy()
temp_df = temp_df.reset_index(drop=True)
temp_df['label'] = temp_df['tfopwg_disp'].isin(['KP', 'CP']).astype('int')


# In[122]:


del temp_df['tfopwg_disp']


# In[125]:


temp_df.label.value_counts()


# In[133]:


columns = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']

for col in columns:
    temp_df[col] = temp_df[col].fillna(temp_df[col].median())


# In[134]:


temp_df.isna().sum()


# In[135]:


new_df_full_train, new_df_test = train_test_split(temp_df, test_size = 0.2, random_state = 42)


# In[136]:


new_df_full_train


# In[137]:


features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad']


# In[138]:


groups = new_df_full_train['tid']
X = new_df_full_train[features]
y = new_df_full_train['label'].values


# In[139]:


del new_df_full_train['label']


# ### Logistic Regression

# In[175]:


gkf = GroupKFold(n_splits=5)

lr_scores = cross_val_score(
    lr_model,
    X,
    y,
    groups=groups,
    cv=gkf,
    scoring="roc_auc",
)


# In[176]:


print(lr_scores)
print("Mean AUC:", lr_scores.mean())
print("Std:", lr_scores.std())


# ## Decision Tree

# In[177]:


gkf = GroupKFold(n_splits=5)

dt_scores = cross_val_score(
    dt_model,
    X,
    y,
    groups=groups,
    cv=gkf,
    scoring="roc_auc",
)


# In[178]:


print(dt_scores)
print("Mean AUC:", dt_scores.mean())
print("Std:", dt_scores.std())


# In[179]:


gkf = GroupKFold(n_splits=5)

rf_scores = cross_val_score(
    rf_model,
    X,
    y,
    groups=groups,
    cv=gkf,
    scoring="roc_auc",
)


# In[180]:


print(rf_scores)
print("Mean AUC:", rf_scores.mean())
print("Std:", rf_scores.std())


# In[183]:


from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    eta=0.1,
    max_depth=10,
    min_child_weight=5,
    objective='binary:logistic',
    n_estimators=200,
    n_jobs=8,
    random_state=1,
    eval_metric='auc'
)


# In[186]:


gkf = GroupKFold(n_splits=5)

xgb_scores = cross_val_score(
    xgb_model,
    X,
    y,
    groups=groups,
    cv=gkf,
    scoring="roc_auc",
)


# In[187]:


print(xgb_scores)
print("Mean AUC:", xgb_scores.mean())
print("Std:", xgb_scores.std())


# In[ ]:




