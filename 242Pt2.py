#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
data = pd.read_csv (r'Downloads/CS242_Proj1_Release/IPPS_DRG_FY2017.csv')

drgDef = data['DRG Definition']
provID = data['Provider Id']
provST = data['Provider State']
totalTD = data['Total Discharges']
avgCC = data['Average Covered Charges']
avgTP = data['Average Total Payments']
avgMP = data['Average Medicare Payments']


# In[18]:


#1.1a
uniProvID = data['Provider Id'].unique()
print("Unique Provider IDs:", 3182)
uniST = data['Provider State'].unique()
newData = pd.DataFrame(columns=uniST)
maxim = 0
mini=0
for j in uniST:
    stRows= data[data['Provider State'] == j]
    num = len(stRows['Provider Id'].unique())
    if num > maxim:
        maxim = num
        fin = j
print ("State with most unique providers:", fin)
#1.1b
me = data['Total Discharges'].mean()
med = data['Total Discharges'].median()
std = data['Total Discharges'].std()
print('mean:',me,'median:', med,'standard deviation:', std)

#1.1c
drg = data['DRG Definition'].unique()
print("unique DRG Defintions:", len(drg))
most = data['DRG Definition'].value_counts().keys()[0]
print('Most common DRG:', most)

#1.1d
mini = 10000000000
nu =[]
for j in drg:
    stRows= data[data['DRG Definition'] == j]
    nu = (stRows['Total Discharges']).sum()
   
    if nu < mini:
        mini = nu
    nu = []
print(mini)
v = data['DRG Definition'][data['Total Discharges'].idxmin()]
print("Least amount of discharges:", v, "at", mini, "discharges")


# In[19]:


#1.2 See R studio


# In[ ]:





# In[64]:


#2.1
# Create a list of lists with tuples as the individual entries. Each list in the main list will correspont to a 
# DRG charge so the first list is all the values of the first drg charge so on. Then each of those lists will contain 
# tuples with (provider id , mean values) which can then be used to create the data frame in 2.2  
drgDef = data['DRG Definition']
inner = []
drgs = []
uniqueDRG = drgDef.unique()
drgVals = drgDef.value_counts().index.tolist()[0:100]
drgVals.sort()
colNames=[]
uniProv = []
state = []
for drg in drgVals:
    colNames.append("DRG Charges " + drg[:3])
    temp = data[data['DRG Definition'] == drg] #dataFrame with only top 100 drgs
    providers = temp['Provider Id'].unique()
    uniProv.append(providers)
    for provider in providers:
        prov1 = temp[temp['Provider Id']== provider]
        mean = (prov1['Average Covered Charges']).mean()
        t = provider, mean
        inner.append(t)
    drgs.append(inner)
    inner = []
finUniProv = [y for x in uniProv for y in x]
finalUniProv = list(set(finUniProv))
finalUniProv.sort()
newData = pd.DataFrame(columns=colNames)
for i in finalUniProv:
    t = data[data['Provider Id']==i]['Provider State']
    state.append(t.iloc[0])
newData.insert(0, 'Prov.Id',finalUniProv)
newData.insert(1, 'Prov. State', state)
print(drgs)


# In[41]:


#2.2
import numpy as np
count = 0
colIdx = 2
colNamesIdx = 0
lis = []
for i in drgs:

    for j in i:
        while j[0] != finalUniProv[count]:
                lis.append(np.NaN)
                count +=1 
        if j[0] == finalUniProv[count]:
            lis.append(j[1])
            count +=1
    while len(lis) < 3169:
        lis.append(np.NaN)
    count =  0
    newData.iloc[:,colIdx] = lis
    colIdx +=1
    lis = []

print(newData)


# In[49]:


#2.3 
#You can either input values for the missing data or you can delete the missing values in the data
#there are no duplicates in the data
i = newData.iloc[:,2:]
h = i.dropna()
v = h.iloc[:,2:].corr()


# In[61]:


#3.1

fullDroppedNAN = newData.dropna()
i = newData.iloc[:,2:]
droppednan = i.dropna()
v = h.iloc[:,2:].corr()

corrMat = v.replace(1, np.nan)

maxVal = 0
maxRow = 0
maxColumn = 0
minVal = 1
minRow = 0
minColumn = 0
for row in corrMat:
    maxCol = corrMat[row].idxmax(axis =1)
    minCol = corrMat[row].idxmin(axis = 1)
    #print(type(maxCol))
    if (corrMat[row][maxCol] > maxVal):
        maxVal = corrMat[row][maxCol]
        maxRow = row
        maxColumn = maxCol
    if (corrMat[row][minCol] < minVal):
        minVal = corrMat[row][minCol]
        minRow = row
        minColumn = minCol
print(corrMat)


# In[57]:


#3.1a
#high positive association
droppednan.plot.scatter(x='DRG Charges 066', y ='DRG Charges 065') #Slope = 1.2
droppednan.plot.scatter(x='DRG Charges 193', y ='DRG Charges 871') #slpoe = 1.3

#low postive asscocation
droppednan.plot.scatter(x='DRG Charges 483', y ='DRG Charges 100') #Slope = 1.2
droppednan.plot.scatter(x='DRG Charges 483', y ='DRG Charges 918') #slpoe = 1.3


# In[58]:


#3.1b
#correlation matrix 
corr1 = corrMat.loc['DRG Charges 066','DRG Charges 065']
corr2 = corrMat.loc['DRG Charges 193','DRG Charges 871']
corr3 = corrMat.loc['DRG Charges 483','DRG Charges 100']
corr4 = corrMat.loc['DRG Charges 483','DRG Charges 918']
print("Correlation between DRG Charges 066 and 065:", corr1)
print("Correlation between DRG Charges 193 and 871:", corr2)
print("Correlation between DRG Charges 483 and 100:", corr3)
print("Correlation between DRG Charges 483 and 918:", corr4)


# In[60]:


#3.2a

boxPlotDF1 = fullDroppedNAN.iloc[:,1:3]
newDF = pd.DataFrame(columns =[ 'DRG Charges 039','Prov. State'] )
AL1 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'AL']
CT1 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'CT']   
MI1 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'MI']
FL1 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'FL']
NY1 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'NY']
TX1 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'TX']
newDF = pd.concat([AL1, CT1, MI1, FL1, NY1, TX1])       

print(newDF.boxplot(column = 'DRG Charges 039', by='Prov. State'))

boxPlotDF1 = fullDroppedNAN.iloc[:,[1, 10]]
newDF = pd.DataFrame(columns =[ 'DRG Charges 101','Prov. State'] )
AL = boxPlotDF1[boxPlotDF1['Prov. State'] == 'AL']
CT = boxPlotDF1[boxPlotDF1['Prov. State'] == 'CT']   
MI = boxPlotDF1[boxPlotDF1['Prov. State'] == 'MI']
FL = boxPlotDF1[boxPlotDF1['Prov. State'] == 'FL']
NY = boxPlotDF1[boxPlotDF1['Prov. State'] == 'NY']
TX = boxPlotDF1[boxPlotDF1['Prov. State'] == 'TX']
newDF = pd.concat([AL, CT, MI, FL, NY, TX])       

print(newDF.boxplot(column = 'DRG Charges 101', by='Prov. State'))

boxPlotDF1 = fullDroppedNAN.iloc[:,[1, 50]]
newDF = pd.DataFrame(columns =[ 'DRG Charges 313','Prov. State'] )
AL2 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'AL']
CT2 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'CT']   
MI2 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'MI']
FL2 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'FL']
NY2 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'NY']
TX2 = boxPlotDF1[boxPlotDF1['Prov. State'] == 'TX']
newDF = pd.concat([AL2, CT2, MI2, FL2, NY2, TX2])       
print(newDF.boxplot(column = 'DRG Charges 313', by='Prov. State'))


# In[62]:


from scipy import stats
#3.2b
#H0: CT and MI have the same means for DRG Charges 313.
#H1: The mean of CT is greater than MI for DRG Charges 313. 
# Performing a one sided paired t-test 
#alpha is .05
t,p = stats.ttest_rel(CT2['DRG Charges 313'], MI2['DRG Charges 313'])
print('test statistic', t, 'p-value',p)

print('p=',p , '> alpha = .05 therefore we fail to reject the null hypothesis that CT and MI have the same means for DRG Charges 313 ')


# In[63]:


from scipy import stats

#3.2c
#H0: NY and TX have the same means for DRG Charges 313, 039 and 101.
#H1: The mean of NY is greater than TX for DRG Charges 313, 039 and 101.
#Performing a one sided paired t-test
NYlis= [] 
TXlis = []
for i in NY1['DRG Charges 039']:
    NYlis.append(i)
for i in NY['DRG Charges 101']:
    NYlis.append(i)
for i in NY2['DRG Charges 313']:
    NYlis.append(i)
    
for i in TX1['DRG Charges 039']:
    TXlis.append(i)
for i in TX['DRG Charges 101']:
    TXlis.append(i)
for i in TX2['DRG Charges 313']:
    TXlis.append(i)


t,p = stats.ttest_rel(TXlis, NYlis)
print('One sided paired ')
print('test statistic', t, 'p-value',p)
print('p=',p , '< alpha = .05 therefore we reject the null so we have significant evidence to lean towards the alternative hypothesis that The mean of NY is greater than TX for DRG Charges 313, 039 and 101  ')
print('One sided unpaired ')
#one sided unpaired
t,p = stats.ttest_ind(TXlis, NYlis)
print('test statistic', t, 'p-value',p)
print('p=',p , '> alpha = .05 therefore we fail to reject the null hypothesis that  The mean of NY is greater than TX for DRG Charges 313, 039 and 101.')
print("The test statistics and the p values changed slightly but the result was significantly different beacuse I used .05 as alpha and my p was less than alpha for my paried test but for unpaired it slightly greater than alpha so it chaged the end result")


# In[ ]:




