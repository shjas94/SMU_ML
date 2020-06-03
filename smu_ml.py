#!/usr/bin/env python
# coding: utf-8

# ## anaconda prompt 열어서 pip install catboost 하고 conda install seaborn 해주면 됩니다

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import catboost as cat
import gc


# In[2]:


# train = pd.read_csv('train_V2_original.csv')
# test = pd.read_csv('test_V2_original.csv')
train = pd.read_pickle('train_V2.pkl')
# test = pd.read_pickle('test_V2.pkl')


# In[3]:


train.head()


# In[4]:


# test.head()


# In[5]:


train.info()


# In[6]:


def rstr(df, pred=None):
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/obs) *100
    skewness = df.skew()
    kurtosis = df.kurt()
    print('Data shape: ', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis', 'corr']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis=1)
    else:
        corr = df.corr()[pred]
        str =pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis=1, sort=False)
        corr_col = 'corr ' + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis',  corr_col]
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str


# ### feature들의 대략적인 통계치, 마지막 열은 target과의 상관계수를 의미 -> 1에 가까울수록 target과의 선형관계가 높다고 볼 수 있음
# ### but 선형관계를 나타내는 지표이기 때문에 낮다고 해서 아예 관계가 없다고 볼 수는 없다.
# #### 데이터 크기 줄인 후 실행하니 일부 지표는 안나옴. 참고

# In[7]:


pd.set_option('display.max_rows', None)
details = rstr(train, 'winPlacePerc')
display(details.sort_values(by='corr winPlacePerc', ascending=False))


# # **데이터 크기 줄이기**
# # 이 부분은 다른 kernel 참조하여 작성했음
# # 최초 실행 이후 주석처리할것!!

# In[9]:


def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df


# In[10]:


# reduced_train = reduce_mem_usage(train)
# reduced_test = reduce_mem_usage(test)
# reduced_train.to_pickle('train_V2.pkl')
# reduced_test.to_pickle('test_V2.pkl')
# del train
# del test
# gc.collect()


# ## EDA

# In[8]:


# 몇번의 매치로부터 추출된 데이터들인가?
train.loc[:, 'matchId'].nunique()


# In[9]:


# 게임 종류(match type)에 따라 나누기
match_type = train.loc[:, 'matchType'].value_counts().to_frame().reset_index()
match_type.columns = ['matchType', 'Count']
match_type


# In[10]:


plt.figure(figsize=(15,8))
type_count = match_type.matchType.values
ax = sns.barplot(x='matchType', y='Count', data=match_type)
ax.set_xticklabels(type_count, rotation=60, fontsize=16)
ax.set_title('Match type')
plt.show()


# In[11]:


match_type2 = train.loc[:, 'matchType'].value_counts().to_frame()
class_squad = match_type2.loc[["squad-fpp","squad","normal-squad-fpp","normal-squad"],"matchType"].sum()
class_duo = match_type2.loc[["duo-fpp","duo","normal-duo-fpp","normal-duo"],"matchType"].sum()
class_solo = match_type2.loc[["solo-fpp","solo","normal-solo-fpp","normal-solo"],"matchType"].sum()
classify = pd.DataFrame([class_squad, class_duo, class_solo], index=['squad', 'duo', 'solo'], columns = ['Count'])
classify

psquad = classify.loc['squad', 'Count']
pduo = classify.loc['duo', 'Count']
psolo = classify.loc['solo', 'Count']
total = psquad + pduo + psolo

print('전체 중에서 스쿼드의 비율은 %.2f, 듀오의 비율은 %.2f, 솔로의 비율은 %.2f입니다.' 
      % (psquad / total, pduo / total, psolo / total))
## 일단 이벤트 매치는 생각 안함


# In[12]:


fig1, ax1 = plt.subplots(figsize=(5, 5))
labels = ['Squad', 'Duo', 'Solo']

wedges, texts, autotexts = ax1.pie(classify['Count'], textprops=dict(color='w'), 
                                   autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
ax1.legend(wedges, labels,
          title='Type',
          loc='center left',
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=12, weight='bold')
plt.show()


# In[13]:


# 한 매치에 몇팀?
sns.kdeplot(train['numGroups'], bw=0.15)

plt.figure(figsize=(15,8))
ax = sns.distplot(train['numGroups'])
ax.set_title('Number of groups')
plt.show()


# ## kills & damagedealt

# In[14]:


plt.figure(figsize=(15,8))
ax1 = sns.boxplot(x='kills', y='damageDealt', data=train)
ax.set_title('Kills & Damage Dealt')
plt.show()
## 단위 통일 후  pca 고려 가능


# In[16]:


# 핵쟁이를 찾아보자.
train[(train['kills'] > 43) & (train['headshotKills']/train['kills'] > 0.5)][
    ['assists', 'damageDealt', 'headshotKills', 'kills', 'longestKill']]


# In[17]:


ride = train.query('rideDistance >0 & rideDistance <10000')
walk = train.query('walkDistance >0 & walkDistance <4000')
ride.hist('rideDistance', bins=50, figsize = (15,10))
walk.hist('walkDistance', bins=50, figsize = (15,10))
plt.show()


# In[18]:


# 총 Distance는 어떤식?
travel_dist = train["walkDistance"] + train["rideDistance"] + train["swimDistance"]
travel_dist = travel_dist[travel_dist<5000]
travel_dist.hist(bins=50, figsize = (15,10))
plt.show()


# ***실제 분포는 walkDistance와 totalDisance가 유사 -> distance 관련 데이터는 합치기!!***

# In[19]:


train['weaponsAcquired'].describe()


# In[20]:


train.hist('weaponsAcquired', figsize=(15, 10), range=(0, 10), rwidth=0.9)
plt.show()


# In[ ]:




