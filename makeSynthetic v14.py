# -*- coding: utf-8 -*-
"""
Synthetic regions
1) Import data and merge to current list of GP practices
2) Run clustering algorithm and select best clusters.
3) Remove vanguard practices from clusters, so their effects don't impact on
expected values
4) Calculate expected rates for all practices in each cluster by month (to
account for seasonality effects), age, sex and deprivation quintiles.
5) For each vanguard, match vanguard to a cluster
6) Extract activity of non-vanguards from chosen cluster.
7) Scale GP practices in cluster by random alpha and calcualte percentage error
from actual across all metrics.


Created on Tue Oct 04 16:03:30 2016
@author: DEVDH102
"""
#Imports
import pandas as pd
import numpy as np
import heapq
from random import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Constants
path = "C:/Users/James Crosbie/Documents/Projects/Synthetic Region/"

#Files
filelist = [ 'gppsbem2015',   # gp patient survey black & ethnic minority 2015
             'GP_AS_pop',     # gp age sex registered population  YY
             'GPimdpoppc1516'  # gp imd population quintiles Sep 2015/16
             ]
#Activities
activitydic = {'Beddays'    : ['TBD1213','TBD1314','TBD1415','TBD1516','TBD1617'],
               'Admissions' : ['EA1213' ,'EA1314' ,'EA1415' ,'EA1516','EA1617']
               }

#vanguard demographic data - for matching vanguards to their clusters
VanguardDemographics = 'VanguardDemographics'

#
#  ** Clustering **
#
#read in files and merge into one file based on current list of gp practices
gplist=pd.read_csv(path+'gp_current2.csv').drop(['Vanguard'],1)
gplist = gplist.rename(columns={'OrgID':'Vanguard'})
for file in filelist:
    gplist = gplist.merge(pd.read_csv( path + 'clustering/' + file + '.csv'),on='GPCode',how='left')

#Clean data
gplist = gplist.replace(to_replace=['*','~','#DIV/0!'], value=0)
gplist = gplist.fillna(0)

#Run clustering algorithm
gpcluster = gplist.drop(['GPCode','IsVanguard','Vanguard'],1)
scalar = StandardScaler().fit(gpcluster)
gpvalues = scalar.transform(gpcluster)

#mylist = []
#for j in range(2,11,1):
#   kmeans = KMeans(n_clusters=j).fit(gpvalues)
#   print(kmeans.score(gpvalues))
#   mylist.append(metrics.silhouette_score(gpvalues, kmeans.labels_, metric='euclidean'))

'''
Cluster analysis shows that the natural number of clusters is 2.
'''

kmeans = KMeans(n_clusters=2).fit(gpvalues)

# Create a PCA model.
pca_ = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_.fit_transform(gpvalues)

# Make a scatter plot, shaded according to cluster assignment.
plt.figure(figsize=(10, 8))
plt.scatter(plot_columns[:, 0], plot_columns[:, 1], c=kmeans.labels_, s=50)
plt.show()


#attach cluster grouping to gp list
cluster = pd.Series(kmeans.labels_,name='Cluster')

gplist =  gplist.reset_index()
gplist = pd.concat([gplist,cluster],axis=1)
gplist = gplist.rename(columns={'OrgID':'Vanguard'})
gplist = gplist[['GPCode','Vanguard','Cluster']]

#write file to CSV
gplist.to_csv(path+'gplist.csv')


#tidy up work area
del filelist, gpcluster, gpvalues, cluster, plot_columns

#
#   ** Make practice level datasets **
#
#import population files
temp = pd.read_csv( path + 'activity/2017_18_Q1_SUS_EA_BD_POP.csv' )
temp = temp.replace({"Oct-14":"10-14", "05-Sep":"5-9", "00-04":"0-4", "05-09":"5-9", "95+":"85+", '90-94':'85+'})
temp = temp.rename(columns={'Activity_Month':'Month', 'AgeBand':'Age',
                            'Pop':'Population', 'EA':'Admissions', 'BD':'Beddays'})
temp = temp.fillna(0)
temp['Age'] = temp['Age'].str.replace(" - ","-")
temp['Age'] = temp['Age'].str.strip()
temp['GPCode'] = temp['GPCode'].str.strip()
temp = temp[ temp['Sex'].isin([1,2]) ]


def makeYear(row):
   if  row['Month'] in range(4,13) :
      return int(row['Financial_Year'][:4])
   else:
      return 2000 + int(row['Financial_Year'][-2:])

temp['Year'] = temp.apply(lambda row: makeYear(row),axis=1)

#tidy up dataset
activity = pd.concat([temp[['Year','Month','Sex']].astype(int),
                      temp[list(['GPCode','Age','Population','Admissions','Beddays'])]],axis=1)


#full activity file - actual activity levels by GP by age & sex
activity = activity.merge(gplist, on='GPCode', how='left')

#Calculate activity levels for the vanguards
vanguard_base = activity[ activity['Vanguard'] != 51 ]
activity_vanguard = activity.groupby(['Vanguard','Year','Month'],as_index=False).sum()
activity_vanguard = activity_vanguard[list(['Vanguard','Year','Month','Admissions','Beddays'])]

#
# ** Expected Rates **
#
# calculate expected rates for non-vanguards by cluster, time-period, age, sex
# merge population and activity dataframes
activity = activity[ (activity['Vanguard'] == 51 ) &
                     (activity['Sex'].isin([1,2])) &
                     (activity['Month'].isin(range(1,13)))]

activity_base = activity.groupby(['Cluster','Year','Month','Sex','Age'],as_index=False).sum()

for _ , i in enumerate(activitydic):
    activity_base['ExpRate_' + i] = activity_base[i] / activity_base['Population']

activity_base = activity_base.drop(['Admissions','Beddays','Population','Vanguard'],1)
activity = activity.merge(activity_base,on=['Cluster','Year','Month','Sex','Age'],how='left')

for _ , i in enumerate(activitydic):
   activity['Expected_' + i] = activity['ExpRate_' + i] * activity['Population']

activity = activity.groupby(['Year','Month','GPCode','Vanguard','Cluster'],as_index=False).sum()
activity = activity.drop(['Sex'],1)
for _ , i in enumerate(activitydic):
   activity = activity.drop(['ExpRate_' + i],1)



#To make synthetic need practives that have full stable time-series over baseline period
#Only keep practices with a full time-series
activity = activity[ activity['Vanguard'] == 51 ]
act_min = activity['GPCode'].value_counts().reset_index()
act_min=act_min.rename(columns={'GPCode':'Count','index':'practice'})
practiceToKeep= act_min[ act_min['Count'] == 63 ]
temp = list(set(practiceToKeep['practice']))
activity = activity[ activity['GPCode'].isin(temp) ]
#Remove small practices from donor pool
act_min = activity.groupby(['GPCode','Year','Month'],as_index=False).sum()
practiceToKeep= act_min[ (act_min['Admissions'] >= 10) & (act_min['Beddays'] >= 50) ]
temp = list(set(practiceToKeep['GPCode']))
activity = activity[ activity['GPCode'].isin(temp) ]
del act_min, practiceToKeep, temp

#
# match each Vanguard to Cluster
#
#Read in vanguard demographic data file and tidy up
vanguards = pd.read_csv( path + 'clustering/' + VanguardDemographics + '.csv')
vanguards = vanguards.replace(to_replace=['*','~','#DIV/0!','nan'], value=0)
#vanguards = vanguards.fillna(0)
#vanguards = vanguards.apply(lambda x: pd.to_numeric(x, errors='ignore'))

#match vanguard to cluster
vanguards2 = vanguards.drop(['ORG_ID'],1)
vanguards_scaled = scalar.transform(vanguards2)
vanguardClusters = pd.Series(kmeans.predict(vanguards_scaled),name='Cluster')
vanguards = pd.concat([vanguards,vanguardClusters],axis=1)
vanguards = vanguards[['ORG_ID','Cluster']]
vanguards = vanguards.rename(columns={'ORG_ID':'ID_Org'})
vanguards = vanguards.sort_values(['ID_Org'])

VanguardList = vanguards['ID_Org'].unique().tolist()
VanguardList = list(map(int, VanguardList))

#tidy up
del VanguardDemographics, activity_base, vanguardClusters, vanguards2

# Write Files to CSV
activity.to_csv(path+'activity.csv')
activity_vanguard.to_csv(path+'vanguardActivity.csv')
vanguards.to_csv(path+'vanguards.csv')

###############################################################################
#
# ** Load datafiles  **
#
gplist = pd.read_csv(path+'gplist.csv')
activity = pd.read_csv(path+'activity.csv')
activity_vanguard = pd.read_csv(path+'vanguardActivity.csv')
vanguards = pd.read_csv(path+'vanguards.csv')

#Put things in the right order
activity = activity.sort_values(by=['GPCode','Year','Month'])
activity_vanguard = activity_vanguard.sort_values(by=['Vanguard','Year','Month'])   #needed for yEA and yTBD

vanguardList = vanguards['ID_Org'].unique().tolist()


#
# Make Synthetic Region
#

def cost(theta, X1, y1, X2, y2):
    penalty = 0.01
    inner1 = np.absolute( (X1.T * theta.T - y1.T) / y1.T)
    inner2 = np.absolute( (X2.T * theta.T - y2.T) / y2.T)
    return np.sum(inner1) + np.sum(inner2) + penalty * np.sum(theta.T)


def anneal(x1, y1, x2, y2, weights):
    old_loss = cost(weights, x1, y1, x2, y2)
    solution = weights
    T = 100.0
    T_min = 0.001
    alpha = 0.9
    while T > T_min:
        counter = 1
        while counter <= 100:
            new_neighbour = neighbour(weights,T)
            new_neighbour = np.matrix(new_neighbour)
            new_loss = cost(new_neighbour, x1, y1, x2, y2)
            ap = acceptance_probability(old_loss, new_loss, T)
            rn = random()
            if ap > rn:
                solution = new_neighbour
                old_loss = new_loss
                try:
                   print "Loss %.2f" %old_loss
                except:
                   print weights
            counter += 1
        T = T * alpha
    return solution, old_loss


def neighbour(v,T):
   v = v.tolist()
   return [np.absolute(v_i + np.random.uniform(-1, 1) * T ) for v_i in v[0]]


def acceptance_probability(o,n,T):
   return np.exp((o-n)/T)


def heapsort(iterable):
   h = []
   for value in iterable:
      heapq.heappush(h, value[1][1])
   return [heapq.heappop(h) for i in range(len(h))]


def makeNonVanguards(nonVanguards, l=36):
   nonVanguardActivity = activity[ activity['GPCode'].isin(nonVanguards) ].drop(list(['Vanguard','Cluster','Population']+activitydic.keys()),1)
   nonVanguardActivity = nonVanguardActivity.sort_values(by=['GPCode','Year','Month'])

   mylist = zip(nonVanguardActivity['Year'][:l].tolist(),nonVanguardActivity['Month'][:l].tolist())
   yearmnthpairs = zip(nonVanguardActivity['Year'].tolist(),nonVanguardActivity['Month'].tolist())
   filter = [True if (i,j) in mylist else False for (i,j) in yearmnthpairs ]

   nonVanguardActivity = nonVanguardActivity[ filter ]

   xEA, xTBD = np.empty((0,l),float),  np.empty((0,l),float)
#   ones = np.array([1 for _ in range(l)])
#   zeros  =np.array([0 for _ in range(l)])

#   xEA = np.append(xEA,[ones],axis=0)
#   xEA = np.append(xEA,[zeros],axis=0)

#   xTBD = np.append(xTBD,[zeros],axis=0)
#   xTBD = np.append(xTBD,[ones],axis=0)

   ClusterSize = len(nonVanguards)
   for j in range(ClusterSize):
      temp = np.array(nonVanguardActivity["Expected_Admissions"][ j* l : (j+1)*l] )
      xEA = np.append(xEA,[temp],axis=0)
      temp = np.array(nonVanguardActivity["Expected_Beddays"][ j* l : (j+1)*l] )
      xTBD = np.append(xTBD,[temp],axis=0)
   return xEA, xTBD


def conv(x):
   return list(np.array(x).reshape(-1,))


def plotTimeSeries(i,yEA,yEA_hat,yTBD,yTBD_hat):
   fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))

   tsLength=len(yEA)

   ax[0].set_xlabel('Baseline Months')
   ax[0].set_ylabel('Emergency Admissions')
   ax[0].set_ylim(0,max(yEA + yEA_hat)*1.2)
   ax[0].set_xlim(1,tsLength)
   ax[0].plot(range(1,tsLength+1),yEA, 'r',label="yEA")
   ax[0].plot(range(1,tsLength+1),yEA_hat, 'b', label="yEA_hat")
   ax[0].axvline(min(tsLength,36),color='black', linestyle='--', linewidth=3)
   ax[0].legend(loc='lower right')

   ax[1].set_xlabel('Baseline Months')
   ax[1].set_ylabel('Total Bed-days')
   ax[1].set_ylim(0,max(yTBD+yTBD_hat)*1.2)
   ax[1].set_xlim(0,tsLength)
   ax[1].plot(range(1,tsLength+1),yTBD, 'm', label='yTBD')
   ax[1].plot(range(1,tsLength+1),yTBD_hat, 'g', label='yTBD_hat')
   ax[1].axvline(min(tsLength,36),color='black', linestyle='--', linewidth=3)
   ax[1].legend(loc='lower right')

   if tsLength == 36:
      text = "BaselinePlot_v"+str(i)
   else:
      text = "FullSeries_v"+str(i)

   plt.savefig(path+'/images/'+ text+'.png')
   plt.show()



def assessFit(yEA, yTBD, xEA, xTBD, mySol):
   yEA_hat = np.dot( np.array(mySol[0][0]),xEA)
   yTBD_hat = np.dot( np.array(mySol[0][0]),xTBD)

   yEA, yTBD=conv(yEA), conv(yTBD)
   yEA_hat, yTBD_hat=conv(yEA_hat), conv(yTBD_hat)

   upLiftEA = sum([yEA[x] - yEA_hat[x] for x in range(36)])/36
   upLiftTBD = sum([yTBD[x] - yTBD_hat[x] for x in range(36)])/36

   yEA_hat = [yEA_hat[x] + upLiftEA for x in range(len(yEA_hat))]
   yTBD_hat = [yTBD_hat[x]+ upLiftTBD  for x in range(len(yTBD_hat))]

   return yEA, yTBD, yEA_hat, yTBD_hat


#Constants for fit
nRuns = 100
fitSize = 1000
placeboSize = 1000

#vanguardList = [2]#,4,12,13]
SyntheticRegion = pd.DataFrame()
fitDF = np.zeros((fitSize+1,len(vanguardList)))
placebo = np.zeros((len(vanguardList),placeboSize,2)) #vanguard, fit before, fit after

for i in vanguardList:
   yEA = np.matrix(activity_vanguard[ activity_vanguard['Vanguard'] == i ]['Admissions'][:48] )
   yTBD = np.matrix(activity_vanguard[ activity_vanguard['Vanguard'] == i ]['Beddays'][:48] )
   sample_size =  gplist['GPCode'][ gplist['Vanguard'] == i ].unique().shape[0]
   cluster = vanguards['Cluster'][ vanguards['ID_Org'] == i].iloc[0]

   for n in range(nRuns):
        try:
            nonVanguards = np.random.choice(activity['GPCode'][ activity['Cluster'] == cluster ].unique().tolist(),size=sample_size,replace=False)
        except:
            sample_size = len(activity['GPCode'][ activity['Cluster'] == cluster ].unique().tolist())
            nonVanguards = np.random.choice(activity['GPCode'][ activity['Cluster'] == cluster ].unique().tolist(),size=sample_size,replace=False)

        nonVanguards = nonVanguards.tolist()
        nonVanguards.sort()

        xEA, xTBD = makeNonVanguards(nonVanguards,48)
        alpha = np.matrix([0 for _ in range(xEA.shape[0])])

        #print xEA.shape, yEA.shape, xTBD.shape, yTBD.shape, alpha.shape
        #print "Initial solution %.2f" %cost(alpha, xEA, yEA, xTBD, yTBD)

        #Using Simulated Annealing
        mySol = anneal(xEA, yEA, xTBD, yTBD, alpha)
        print "Synthetic. Vanguard {i}. Iteration {n}. Final solution {s:{prec}} ".format(i=i,n=n,s=mySol[1], prec=0.5)
        print "#"*50

        if n == 0:
            best_solution = mySol, nonVanguards
        elif mySol[1] < best_solution[0][1] :
            best_solution = mySol, nonVanguards

   print
   print " End of Simulated Annealing "
   print
   mySol = best_solution
   print "Best solution {}".format(mySol[0][1])

   yEA, yTBD, yEA_hat, yTBD_hat = assessFit(yEA, yTBD, xEA, xTBD, mySol)
   plotTimeSeries(i,yEA,yEA_hat,yTBD,yTBD_hat)
   fitDF[0,vanguardList.index(i)] = (np.sum([ np.absolute(k-j)/k for k,j in zip(yEA,yEA_hat) ])
                   + np.sum([ np.absolute(k-j)/k for k,j in zip(yTBD,yTBD_hat) ]))/36

#Add to SyntheticRegion DF
   practices = mySol[1]
   weights = [ k for k in np.array(mySol[0][0]).reshape(-1,) ]
   tmp = pd.DataFrame({'V'+str(i)+'_Practices': practices,
                       'V'+str(i)+'_Weights': weights },
                                index=range(len(practices)))
   SyntheticRegion = pd.concat([SyntheticRegion,tmp],axis=1)

# make full time series
   yEA = np.matrix(activity_vanguard[ activity_vanguard['Vanguard'] == i ]['Admissions'] )
   yTBD = np.matrix(activity_vanguard[ activity_vanguard['Vanguard'] == i ]['Beddays'] )
   xEA, xTBD = makeNonVanguards(practices, yEA.shape[1])
   yEA, yTBD, yEA_hat, yTBD_hat = assessFit(yEA, yTBD, xEA, xTBD, mySol)
   plotTimeSeries(i,yEA,yEA_hat,yTBD,yTBD_hat)

#
#  **  Assess fit **
#
   for k in range(fitSize):
      sample_size =  gplist['GPCode'][ gplist['Vanguard'] == i ].unique().shape[0]
      yEA = np.matrix(activity_vanguard[ activity_vanguard['Vanguard'] == i ]['Admissions'][:36] )
      yTBD = np.matrix(activity_vanguard[ activity_vanguard['Vanguard'] == i ]['Beddays'][:36])

      donor = np.random.choice(activity['GPCode'].unique().tolist(), size=sample_size, replace=False)
      xEA, xTBD = makeNonVanguards(donor,36)
      alpha = np.matrix([0 for _ in range(xEA.shape[0])])

      print "Initial solution `%.2f" %cost(alpha, xEA, yEA, xTBD, yTBD)
      mySol = anneal(xEA, yEA, xTBD, yTBD, alpha)
      print "Fit. Vanguard {i}. Iteration {k}. Final solution {s:{prec}}".format(i=i,k=k,s=mySol[1],prec='.5')
      print "#"*50

      yEA, yTBD, yEA_hat, yTBD_hat = assessFit(yEA, yTBD, xEA, xTBD, mySol)
      fitDF[k+1,vanguardList.index(i)] = (np.sum([ np.absolute(l-j)/l for l,j in zip(yEA,yEA_hat) ]) +
                      np.sum([ np.absolute(l-j)/l for l,j in zip(yTBD,yTBD_hat) ]))/36
#
#  **  Placebo test  **
#
   for k in range(placeboSize):
      sample_size =  gplist['GPCode'][ gplist['Vanguard'] == i ].unique().shape[0]

      placeboVanguard = np.random.choice(activity['GPCode'].unique().tolist(),size=sample_size,replace=False)

      tmp = activity[ activity['GPCode'].isin(placeboVanguard) ][['GPCode','Year','Month','Admissions']]
      tmp = tmp.groupby(['Year','Month'],as_index=False).sum()
      tmp2 = tmp[:36]
      yEA = np.matrix(tmp2.drop(['Year','Month'],axis=1)).transpose()

      tmp = activity[ activity['GPCode'].isin(placeboVanguard) ][['GPCode','Year','Month','Beddays']]
      tmp = tmp.groupby(['Year','Month'],as_index=False).sum()
      tmp2 = tmp[:36]
      yTBD = np.matrix(tmp2.drop(['Year','Month'],axis=1)).transpose()

      donorPool = list(set(activity['GPCode'].unique().tolist()) - set(placeboVanguard))

      #fit
      donor = np.random.choice(donorPool,size=sample_size,replace=False)
      xEA, xTBD = makeNonVanguards(donor, 36)
      alpha = np.matrix([0 for _ in range(xEA.shape[0])])

      print "Initial solution %.2f" %cost(alpha, xEA, yEA, xTBD, yTBD)
      mySol = anneal(xEA, yEA, xTBD, yTBD, alpha)
      print "Placebo.  Vanguard {i}. Iteration {k}. Final solution {s:{prec}}".format(i=i,k=k, s=mySol[1],prec='.5')
      print "#"*50

    #fit before
      yEA, yTBD, yEA_hat, yTBD_hat = assessFit(yEA, yTBD, xEA, xTBD, mySol)
      placebo[vanguardList.index(i),k,0]= (np.sum([ np.absolute(l-j)/l for l,j in zip(yEA,yEA_hat) ])
                              + np.sum([ np.absolute(l-j)/l for l,j in zip(yTBD,yTBD_hat) ]))/36

    #fit after
      yEA = np.matrix(tmp.drop(['Year','Month'],axis=1)).transpose()
      yTBD = np.matrix(tmp.drop(['Year','Month'],axis=1)).transpose()
      xEA, xTBD = makeNonVanguards(donor, yEA.shape[1])

      yEA, yTBD, yEA_hat, yTBD_hat = assessFit(yEA, yTBD, xEA, xTBD, mySol)
      placebo[vanguardList.index(i),k,1]= (np.sum([ np.absolute(l-j)/l for l,j in zip(yEA,yEA_hat) ])
                      + np.sum([ np.absolute(l-j)/l for l,j in zip(yTBD,yTBD_hat) ]))/24
#End


# Write to CSV
SyntheticRegion.to_csv(path+'SyntheticRegion_48period.csv')
with open(path+"placebo.txt", "a") as text_file:
    text_file.write('#Placebo Effects: (vanguard, before, after) {0}\n'.format(placebo.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for k, data_slice in enumerate(placebo):
        text_file.write('# Vanguard {} Before, After\n'.format(vanguardList[k]))
        np.savetxt(text_file,data_slice)



###############################################################################
#
#  **  Analysis of fit and placebo  **
#
for i in vanguardList:
    Vname = vanguards['Vanguard'][ vanguards['ID_Org'] == i].iloc[0]
    plt.title('Fit for vanguard: \n' + str(Vname))
    plt.xlabel('Fit Range')
    plt.ylabel('Frequency')
    plt.hist(fitDF[:,vanguardList.index(i)],bins=50)
    plt.axvline(fitDF[0,vanguardList.index(i)],color='black', linestyle='--', linewidth=3)
    plt.savefig(path+'/images/'+'FitHist_v'+str(i)+'.png')
    plt.show()

    plt.title('Placebo Effects for vanguard: \n' + str(Vname))
    plt.xlabel('Placebo Range')
    plt.ylabel('Frequency')
    plt.hist(placebo[vanguardList.index(i),:,0],bins=50,label="Before",color='blue')
    plt.hist(placebo[vanguardList.index(i),:,1],bins=50,label="After",color='green')
    plt.legend()
    plt.savefig(path+'/images/'+'PlaceboHist_v'+str(i)+'.png')
    plt.show()
