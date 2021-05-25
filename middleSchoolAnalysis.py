# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:48:29 2021

@author: Sangwon Baek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

#Loading the data using panda
schoolData = pd.read_csv('middleSchoolData.csv')

#Finding the number of missing data: 
schoolData.isna().sum()

#Imputing the missing data with the mean of the values 
schoolData['per_pupil_spending']= schoolData['per_pupil_spending'].fillna(value=schoolData['per_pupil_spending'].mean())
schoolData['avg_class_size']= schoolData['avg_class_size'].fillna(value=schoolData['avg_class_size'].mean())
schoolData['asian_percent']= schoolData['asian_percent'].fillna(value=schoolData['asian_percent'].mean())
schoolData['black_percent']= schoolData['black_percent'].fillna(value=schoolData['black_percent'].mean())
schoolData['hispanic_percent']= schoolData['hispanic_percent'].fillna(value=schoolData['hispanic_percent'].mean())
schoolData['multiple_percent']= schoolData['multiple_percent'].fillna(value=schoolData['multiple_percent'].mean())
schoolData['white_percent']= schoolData['white_percent'].fillna(value=schoolData['white_percent'].mean())
schoolData['rigorous_instruction']= schoolData['rigorous_instruction'].fillna(value=schoolData['rigorous_instruction'].mean())
schoolData['collaborative_teachers']= schoolData['collaborative_teachers'].fillna(value=schoolData['collaborative_teachers'].mean())
schoolData['supportive_environment']= schoolData['supportive_environment'].fillna(value=schoolData['supportive_environment'].mean())
schoolData['effective_school_leadership']= schoolData['effective_school_leadership'].fillna(value=schoolData['effective_school_leadership'].mean())
schoolData['strong_family_community_ties']= schoolData['strong_family_community_ties'].fillna(value=schoolData['strong_family_community_ties'].mean())
schoolData['trust']= schoolData['trust'].fillna(value=schoolData['trust'].mean())
schoolData['disability_percent']= schoolData['disability_percent'].fillna(value=schoolData['disability_percent'].mean())
schoolData['poverty_percent']= schoolData['poverty_percent'].fillna(value=schoolData['poverty_percent'].mean())
schoolData['ESL_percent']= schoolData['ESL_percent'].fillna(value=schoolData['ESL_percent'].mean())
schoolData['school_size']= schoolData['school_size'].fillna(value=schoolData['school_size'].mean())
schoolData['student_achievement']= schoolData['student_achievement'].fillna(value=schoolData['student_achievement'].mean())
schoolData['reading_scores_exceed']= schoolData['reading_scores_exceed'].fillna(value=schoolData['reading_scores_exceed'].mean())
schoolData['math_scores_exceed']= schoolData['math_scores_exceed'].fillna(value=schoolData['math_scores_exceed'].mean())

#Converting 6 school climate variables and 3 objective achievement variables into numpy array
predictors = schoolData[["applications", "per_pupil_spending", "avg_class_size", "asian_percent", "black_percent", "hispanic_percent", "multiple_percent",
                         "white_percent", "rigorous_instruction", "collaborative_teachers", "supportive_environment", "effective_school_leadership", 
                         "strong_family_community_ties", "trust", "disability_percent", "poverty_percent", "ESL_percent", "school_size"]].to_numpy()
objectiveAchievement = schoolData[["student_achievement", "reading_scores_exceed", "math_scores_exceed"]].to_numpy()
admission = schoolData[["acceptances"]].to_numpy()
schoolDataNP = schoolData.drop(columns=['school_name','dbn']).to_numpy()

#Correlation map of school data
r1 = np.corrcoef(predictors,rowvar=False)
plt.imshow(r1)
plt.colorbar()
plt.title('Correlation Matrix: Predictors')

#Correlation map of school data
r2 = np.corrcoef(objectiveAchievement,rowvar=False)
plt.imshow(r2)
plt.colorbar()
plt.title('Correlation Matrix: Objective Achievement')

#Running PCA on Predictors
pca = PCA()
zscoredSchoolData=stats.zscore(predictors, nan_policy='omit')
pca.fit(zscoredSchoolData)
eigValues = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredSchoolData)

#Running PCA on objectiveAchievement
pca1 = PCA()
zscoredSchoolData1=stats.zscore(objectiveAchievement, nan_policy='omit')
pca1.fit(zscoredSchoolData1)
eigValues1 = pca1.explained_variance_
loadings1 = pca1.components_
rotatedData1 = pca1.fit_transform(zscoredSchoolData1)

#Running PCA on admission
pca2 = PCA()
zscoredSchoolData2=stats.zscore(admission, nan_policy='omit')
pca2.fit(zscoredSchoolData2)
eigValues2 = pca2.explained_variance_
loadings2 = pca2.components_
rotatedData2 = pca2.fit_transform(zscoredSchoolData2)

#Scree plot for predictors with the Kaiser crietrion line
numClasses=18
plt.plot(eigValues)
plt.title('Scree Plot: predictors')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalues')
plt.xticks(np.arange(0,19,step=1))
plt.plot([0,numClasses],[1,1],color='red',linewidth=1.5)

#Scree plot for objective achievement with the Kaiser crietrion line
plt.plot(eigValues1)
plt.title('Scree Plot: objective achievement')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalues')
plt.xticks(np.arange(0,3,step=1))
plt.plot([0,3],[1,1],color='red',linewidth=1.5)

#Factor 1 - predictors
plt.bar(np.linspace(1,numClasses, numClasses), loadings[0,:])
plt.title('Factor 1: predictors')
plt.xlabel('Variables')
plt.ylabel('Loading')
plt.xticks(np.arange(0,19,step=1))

#Factor 2 - predictors
plt.bar(np.linspace(1,numClasses, numClasses), loadings[1,:])
plt.title('Factor 2: predictors')
plt.xlabel('Variables')
plt.ylabel('Loading')
plt.xticks(np.arange(0,19,step=1))

#Factor 3 - predictors
plt.bar(np.linspace(1,numClasses, numClasses), loadings[2,:])
plt.title('Factor 3: predictors')
plt.xlabel('Variables')
plt.ylabel('Loading')
plt.xticks(np.arange(0,19,step=1))

#Factor 4 - predictors
plt.bar(np.linspace(1,numClasses, numClasses), loadings[3,:])
plt.title('Factor 4: predictors')
plt.xlabel('Variables')
plt.ylabel('Loading')
plt.xticks(np.arange(0,19,step=1))

#Factor 1-objectiveAchievement
plt.bar(np.linspace(1,3, 3), loadings1[0,:])
plt.title('Factor 1: objective achievement')
plt.xlabel('Variables')
plt.ylabel('Loading')
plt.xticks(np.arange(0,4,step=1))

#Scatter Plot of Student school perception vs. Objective achievement of school
plt.plot(rotatedData[:,0],rotatedData1[:,0],'o',markersize=2)
plt.xlabel('Student school perception')
plt.ylabel('Objective achievement of school')

#Scatter Plot of Applications vs. Acceptances
plt.plot(schoolData["applications"], schoolData['acceptances'], 'o', markersize=3)
plt.xlabel('Applications')
plt.ylabel('Acceptances')

#Calculation of application rate to identify the better predictor
schoolData2 = pd.DataFrame(data= schoolData, columns=["school_name","applications","acceptances", "school_size"])
schoolData2['applicationRate'] = schoolData2['applications'].divide(schoolData2['school_size'])*100
CorrNumberRateApp = schoolData2.corr(method= 'pearson')

#Calculation of admission proportion per school to identify the school that has the best per student odds
schoolData2['admissionProportionPerSchool'] = schoolData2['acceptances'].divide(schoolData2['applications'])
#replace Nan with 0 : handling NaN values created by 0/0 computation.
schoolData2['admissionProportionPerSchool'] = schoolData2['admissionProportionPerSchool'].fillna(0)

#Convert School size to categorical variables 0=small size, 1=big size
#Determining factor for conversion was the median value of the school size
medianSchoolSize = schoolData["school_size"].median()
schoolSizeData = pd.DataFrame(data=schoolData,columns=["school_size","acceptances"]).to_numpy()
#Converting school size data into small/large categorical variable
for i in range (len(schoolSizeData)):
    if schoolSizeData[i][0] < medianSchoolSize:
        schoolSizeData[i][0]=int(0)
    else:
        schoolSizeData[i][0]=int(1)
#Sorted the data based on small/large, first half is small, second half is large 
schoolSizeData.sort(axis=0)
smallSchoolData, largeSchoolData = np.vsplit(schoolSizeData,2)
#Null hypothesis: the school size affects the number of acceptances that more will get accepted when school size is considered huge.
#We wish to show if this null hypothesis is true or not through the two sample t-test.
t_statistic1, p_value1 = stats.ttest_ind(smallSchoolData[:,1],largeSchoolData[:,1], equal_var=False)

#Convert per spending to categorical variables 0=less spending, 1=more spending
#Determining factor for conversion was the median value of the per student spending
medianPerSpending = schoolData["per_pupil_spending"].median()
perSpendingData = pd.DataFrame(data=schoolData,columns=["per_pupil_spending","acceptances"]).to_numpy()
#Converting per spending data into less/more categorical variable
for i in range (len(perSpendingData)):
    if perSpendingData[i][0] < medianPerSpending:
        perSpendingData[i][0]=int(0)
    else:
        perSpendingData[i][0]=int(1)
#Sorted the data based on less/more, first half is less, second half is more 
perSpendingData.sort(axis=0)
lessSpending, moreSpending = np.vsplit(perSpendingData,2)
#Null hypothesis: the per stduent spending affects the number of acceptances that more will get accepted when spending is more per student.
#We wish to show if this null hypothesis is true or not through the two sample t-test.
t_statistic2, p_value2 = stats.ttest_ind(lessSpending[:,1],moreSpending[:,1], equal_var=False)

#Convert Class size to categorical variables 0=small size, 1=big size
#Determining factor for conversion was the median value of the class size
medianClassSize = schoolData["avg_class_size"].median()
classSizeData = pd.DataFrame(data=schoolData,columns=["avg_class_size","acceptances"]).to_numpy()
#Converting class size data into small/large categorical variable
for i in range (len(classSizeData)):
    if classSizeData[i][0] < medianSchoolSize:
        classSizeData[i][0]=int(0)
    else:
        classSizeData[i][0]=int(1)
#Sorted the data based on small/large, first half is small, second half is large 
classSizeData.sort(axis=0)
smallClassData, largeClassData = np.vsplit(classSizeData,2)
#Null hypothesis: the school size affects the number of acceptances that more will get accepted when school size is considered huge.
#We wish to show if this null hypothesis is true or not through the two sample t-test.
t_statistic3, p_value3 = stats.ttest_ind(smallClassData[:,1],largeClassData[:,1], equal_var=False)

#Create a bar plot for the number of students acceptances in a decreasing rank order.
acceptanceData = pd.DataFrame(data=schoolData,columns=["school_name","acceptances"])
acceptanceTemp = acceptanceData.sort_values(by='acceptances',ascending=False).reset_index()
acceptanceData = acceptanceTemp.drop(columns=['index'])
acceptanceDataNP = acceptanceData.to_numpy()
plt.bar(acceptanceData["school_name"],acceptanceData["acceptances"])
plt.xlabel("school name")
plt.ylabel("acceptances")
plt.title("The number of acceptances per school")
plt.show()

#Total number of student accepted to HSPHS
totalAcceptance = schoolData["acceptances"].sum()
#90% of total acceptance
ninetyAcceptance = totalAcceptance*.9

#indexCount representst the number of school that add up to 90% of the all students acceptance
sumA=0
indexCount=0
for y in range (len(acceptanceData)-1):
    if sumA < ninetyAcceptance:
        sumA += acceptanceDataNP[y][1]
        indexCount+=1
#Proportion of schools accounting 90% of student acceptancce
schoolProportion = indexCount/len(schoolData["acceptances"])
    
#Applying clustering method to idetnfiy what school characteristics are most important for: 
#a) sending students to HSPHS b) improve objective measure of achievement
#the predictors vs. objective achievement
X = np.transpose(np.array([rotatedData[:,0],rotatedData1[:,0]]))
X1 = np.transpose(np.array([rotatedData[:,0],rotatedData2[:,0]]))
numClusters = 9 
Box = np.empty([numClusters,1]) 
Box[:] = np.NaN 

# Compute kMeans clustering: objective measures of achievement and predictors
for i in range(2, 11): 
    kMeans = KMeans(n_clusters = int(i)).fit(X)
    cId = kMeans.labels_ 
    cCoords = kMeans.cluster_centers_
    silhouette = silhouette_samples(X,cId) 
    Box[i-2] = sum(silhouette) 
    plt.subplot(3,3,i-1) 
    plt.hist(silhouette,bins=20) 
    plt.xlim(-0.2,1.5)
    plt.ylim(0,100)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Box[i-2]))) 
#Plotting of the silhouette values vs. the number of clusters
plt.plot(np.linspace(2,10,numClusters),Box)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')    
iVector1 = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for i in iVector1:
    plotIndex = np.argwhere(cId == int(i-1))
    plt.plot(rotatedData[plotIndex,0],rotatedData1[plotIndex,0],'o',markersize=5)
    plt.plot(cCoords[int(i-1),0],cCoords[int(i-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Predictors')
    plt.ylabel('Objective measures of achievement')

# Compute kMeans clustering: admission and predictors
for i in range(2, 11): 
    kMeans = KMeans(n_clusters = int(i)).fit(X1)
    cId = kMeans.labels_ 
    cCoords = kMeans.cluster_centers_
    silhouette = silhouette_samples(X1,cId) 
    Box[i-2] = sum(silhouette) 
    plt.subplot(3,3,i-1) 
    plt.hist(silhouette,bins=20) 
    plt.xlim(-0.2,1.5)
    plt.ylim(0,100)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Box[i-2]))) 
#Plotting of the silhouette values vs. the number of clusters
plt.plot(np.linspace(2,10,numClusters),Box)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')    
iVector2 = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for i in iVector2:
    plotIndex = np.argwhere(cId == int(i-1))
    plt.plot(rotatedData[plotIndex,0],rotatedData2[plotIndex,0],'o',markersize=5)
    plt.plot(cCoords[int(i-1),0],cCoords[int(i-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Predictors')
    plt.ylabel('Objective measures of achievement')


