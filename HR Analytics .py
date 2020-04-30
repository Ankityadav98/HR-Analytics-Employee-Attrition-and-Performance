#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv',index_col='EmployeeNumber')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(),cmap='viridis',cbar=False, yticklabels=False);


# In[7]:


df.corr()


# In[8]:


for col in df.columns:
    print("{}:{}".format(col,df[col].nunique()))
    print("=======================================")


# In[9]:


df.drop(columns=['Over18','StandardHours','EmployeeCount'],inplace=True)


# In[10]:


df['Attrition']=df['Attrition'].map({'Yes':1, 'No':0})


# In[11]:


categorical_col=[]
for col in df.columns:
    if df[col].dtype== object and df[col].nunique()<=50:
        categorical_col.append(col)
print(categorical_col)


# In[12]:


for col in categorical_col:
    print("{}:\n{}".format(col,df[col].value_counts()))
    print("=======================================")


# In[13]:


df.columns


# In[14]:


sns.countplot(x='Attrition',data=df)


# In[15]:


sns.countplot(x='Attrition',hue='PerformanceRating',data=df)


# In[16]:


sns.countplot(x='Attrition',hue='JobInvolvement',data=df)


# In[17]:


sns.scatterplot(x='Age',y='MonthlyIncome',data=df)


# In[18]:


sns.kdeplot(df['Age'],df['MonthlyIncome'],shade=True,cbar=True)


# In[166]:


plt.figure(figsize=(18,12))
sns.heatmap(df.corr(),cmap='RdYlGn',annot=True,fmt='.2f')


# 1.Self relation ie of a feature to itself is equal to 1 as expected.
# 
# 2.JobLevel is highly related to Age as expected as aged employees will generally tend to occupy higher positions in the company.
# 
# 3.MonthlyIncome is very strongly related to joblevel as expected as senior employees will definately earn more.
# 
# 4.PerformanceRating is highly related to PercentSalaryHike which is quite obvious.
# 
# 5.Also note that TotalWorkingYears is highly related to JobLevel which is expected as senior employees must have worked for a larger span of time.
# 
# 6.YearsWithCurrManager is highly related to YearsAtCompany.
# 
# 7.YearsAtCompany is related to YearsInCurrentRole.

# In[20]:


df.corr()['Attrition'].sort_values(ascending=False)


# In[21]:


sns.set(font_scale=2)
plt.figure(figsize=(30,30))
for i,col in enumerate(categorical_col,1):
    plt.subplot(3,3,i)
    sns.barplot(x=f"{col}",y='Attrition',data=df)
    plt.xticks(rotation=90)
plt.tight_layout()


# In[22]:


sns.set(font_scale=1)
sns.boxplot(x='JobRole',y='MonthlyIncome',data=df)
plt.xticks(rotation=90);


# In[23]:


sns.boxplot(x='EducationField',y='MonthlyIncome',data=df)
plt.xticks(rotation=90);


# In[24]:


sns.violinplot(x='EducationField',y='MonthlyIncome',data=df,hue='Attrition',color='Yellow',split=True)
plt.legend(bbox_to_anchor=(1.2,0.65))
plt.xticks(rotation=45);


# In[25]:


plt.subplots(figsize=(15,5))
sns.countplot(x='TotalWorkingYears',data=df)


# In[26]:


plt.figure(figsize=(6,6))
plt.pie(df['EducationField'].value_counts(),labels=df['EducationField'].value_counts().index,autopct='%.2f%%');


# In[27]:


df['EducationField'].value_counts()


# In[28]:


df.groupby(by='JobRole')["PercentSalaryHike","YearsAtCompany","TotalWorkingYears","YearsInCurrentRole","WorkLifeBalance"].mean()


# In[29]:


plt.figure(figsize=(6,6))
plt.pie(df['JobRole'].value_counts(),labels=df['JobRole'].value_counts().index,autopct='%.2f%%');
plt.title('Job Role Distribution',fontdict={'fontsize':22});


# In[30]:


plt.figure(figsize=(14,5))
sns.countplot(x='Age',data=df)


# In[31]:


sns.barplot(x='Education',y='MonthlyIncome',hue='Attrition',data=df)
plt.legend(bbox_to_anchor=(1.2,0.6))


# In[32]:


sns.barplot(y='DistanceFromHome',x='JobRole',hue='Attrition',data=df,dodge=False,alpha=0.4,palette='twilight')
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor=(1.2,0.6));


# ENCODING THE CATEGORICAL COLUMNS.

# In[117]:


from sklearn.preprocessing import LabelEncoder


# In[118]:


le= LabelEncoder()


# In[35]:


for col in categorical_col:
    df[col]=le.fit_transform(df[col])


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


data= df.copy()


# In[38]:


X= data.drop('Attrition',axis=1)
y=data['Attrition']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# USING TREE DECISION METHOD FOR PREDICTION.

# In[ ]:





# In[41]:


from sklearn.tree import DecisionTreeClassifier


# In[42]:


model= DecisionTreeClassifier()


# In[43]:


model.fit(X_train,y_train)


# In[44]:


pred= model.predict(X_test)


# In[45]:


from sklearn.metrics import classification_report,confusion_matrix


# In[46]:


print(classification_report(y_test,pred))


# ACCURACY USING DECISION TREE IS 77%. AND THE CONFUSION MATRIX COMES OUT TO BE.

# In[120]:


print(confusion_matrix(y_test,pred))


# NOW, WE WILL BE TUNING THE HYPERPARAMETERS OF DECISION TREE USING RANDOMIZED SEARCH CROSS VALIDATION
# METHOD FOR IMPROVING THE ACCURACY OF THE MODEL.

# In[55]:


from sklearn.model_selection import RandomizedSearchCV


# In[73]:


params={"criterion":("gini", "entropy"),
        "splitter":("best", "random"), 
        "max_depth":(list(range(1, 20))), 
        "min_samples_split":[2, 3, 4], 
        "min_samples_leaf":list(range(1, 20))}


# In[ ]:





# In[74]:


tree_random= RandomizedSearchCV(model,params,n_iter=100,n_jobs=-1,cv=3,verbose=2)


# In[75]:


tree_random.fit(X_train,y_train)


# In[76]:


tree_random.best_estimator_


# In[79]:


model=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='random')


# In[80]:


model.fit(X_train,y_train)
pred=model.predict(X_test)


# In[81]:


print(classification_report(y_test,pred))


# WE CAN SEE THAT WE HAVE IMPROVED THE ACCURACY FOR DECISION TREE TO 85% USING RANDOM SEARCH CV METHOD. AND THE CONFUSION MATRIX IS FOUND TO BE.

# In[82]:


print(confusion_matrix(y_test,pred))


# In[ ]:





# NOW TRYING TO MAKE A MODEL USING RANDOM FOREST CLASSIFIER

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[85]:


rfc= RandomForestClassifier(n_estimators=100)


# In[86]:


rfc.fit(X_train,y_train)


# In[87]:


rfc_pred= rfc.predict(X_test)


# In[88]:


print(classification_report(y_test,rfc_pred))


# In[89]:


print(confusion_matrix(y_test,rfc_pred))


# Accuracy of this model is 87%.

# In[ ]:




