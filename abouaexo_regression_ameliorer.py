#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os, pandas as pd, numpy as np, matplotlib.pyplot as pl, pickle as pkl
from sklearn import model_selection as modele, linear_model as lm, metrics as mes
import matplotlib as plt
import seaborn as sb


# In[27]:


os.getcwd()


# In[28]:


os.chdir("C:\\Users\\HPPC\\regession logistic")


# In[29]:


data = pd.read_excel("null.xlsx")
data.head()


# In[30]:


df = data.copy()
print(df)


# In[31]:


df.duplicated().sum()


# In[32]:


df.nunique()


# In[33]:


df.isna().sum()


# In[38]:


quant = ['AGE', 'PAR', 'CHOLESTEROL', 'FCMAX', 'DEPRESSION ']
#df.rename('DEPRESSION')


# In[52]:


def recoder(df):
    for i in df.select_dtypes('object').columns:
        df[i]=df[i].astype('category').cat.codes
    return(df)
recoder(df)


# In[53]:


df[quant] = round(df[quant]/df[quant].mean(), 2)


# In[54]:


df.head()


# In[55]:


pl.figure(tight_layout = True, figsize = (14, 6))
pl.suptitle("LES HISTOGRAMMES DES VARIABLES QUANTITATIVES")
for y,x in enumerate(quant):
    pl.subplot(2,5,y+1)
    pl.hist(df[x])
    pl.title(f"{quant[y]}")
pl.show()


# In[56]:


qual = ['SEXE', 'TDT', 'GAJ', 'ECG', 'ANGINE', 'PENTE', 'CŒUR'] # la liste des variables qualitatives

pl.figure(tight_layout = True, figsize = (15,8))
pl.suptitle("LES DIAGRAMMES CIRCULAIRE DES VARIABLES QUALITATIVES")
for y,x in enumerate(qual):
    eff = df[x].value_counts()
    modalite = df[x].unique()
    pl.subplot(2,4,y+1)
    pl.pie(eff, labels = modalite, autopct = '%1.1f%%')
    pl.legend(bbox_to_anchor = (0, 1))
    pl.title(f"{qual[y]}")
pl.show()


# In[57]:


X = df.iloc[:, 0:11]
X


# In[58]:


y = data.iloc[:,11:]
y


# In[59]:


X_train, X_test, Y_train, Y_test = modele.train_test_split(X, y, test_size = 0.2, random_state=1)


# In[60]:


mod = lm.LogisticRegression(penalty = 'none', solver = 'newton-cg')
mod.fit(X_train, Y_train)


# In[61]:


mod.predict_proba(X_test)


# In[62]:


pred = mod.predict(X_test)

pred


# In[63]:


mes.confusion_matrix(Y_test, pred)


# In[64]:


mes.accuracy_score(Y_test, pred)


# In[65]:


mes.recall_score(Y_test, pred)


# In[66]:


mes.precision_score(Y_test, pred)


# In[67]:


# sauvegarde du modele de prediction mod

pkl.dump(mod, open('mod.pkl', 'wb' ))


# In[ ]:




