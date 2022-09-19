from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from random import sample, seed
from numpy.random import uniform
from math import isnan
import matplotlib.cm as cm
import requests
from forbiddenfruit import curse
from pandas_flavor import register_dataframe_method,register_series_method
from IPython.core.magic import register_line_magic, register_cell_magic,register_line_cell_magic
import warnings
from sklearn.metrics import silhouette_samples,silhouette_score


def CheckForClusteringTendencyWithHopkins(X,random_state=42):    
    """
    taken from https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/
    X:numpy array or dataframe
    the closer to 1, the higher probability of clustering tendency    
    X must be scaled priorly.
    """
        
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    if type(X)==np.ndarray:
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    else:
        nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
    seed(random_state) 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        #-------------------bi ara random state yap----------
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        if type(X)==np.ndarray:
            w_dist, _ = nbrs.kneighbors(X[rand_X[j]].reshape(1, -1), 2, return_distance=True)
        else:
            w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H    

def draw_elbow(ks,data):
    wcss = []
    for i in ks:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) 
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(ks, wcss)
    plt.title('Elbow Method')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.xticks(ks)
    plt.show()
    
def biplot(score,coeff,y,variance,labels=None):
    """
    PCA biplot.
    Found at https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
    """
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{},Variance:{}".format(1,variance[0]))
    plt.ylabel("PC{},Variance:{}".format(2,variance[1]))
    plt.grid()
    
    
def drawEpsilonDecider(data,n):
    """
    for DBSCAN
    n: # of neighbours
    data:numpy array
    """
    neigh = NearestNeighbors(n_neighbors=n)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.ylabel("eps")
    plt.xlabel("number of data points")    
    plt.plot(distances) 
      
    
    
def draw_sihoutte(range_n_clusters,data,isbasic=True,printScores=True,random_state=42):
    """
    - isbasic:if True, plots scores as line chart whereas false, plots the sihoutte chart.
    - taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html and modified as needed.
    """
    if isbasic==False:
        silhouette_max=0
        for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(12,4)

            ax1.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = clusterer.fit_predict(data)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
            if silhouette_avg>silhouette_max:
                silhouette_max,nc=silhouette_avg,n_clusters
            print("For n_clusters =", n_clusters,
                "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                        "with n_clusters = %d" % n_clusters),
                        fontsize=14, fontweight='bold')            
            plt.show()
        print(f"Best score is {silhouette_max} for {nc}")
    else:
        ss = []
        for n in range_n_clusters:
            kmeans = KMeans(n_clusters=n, random_state=random_state)
            kmeans.fit_transform(data)
            labels = kmeans.labels_
            score = silhouette_score(data, labels)
            ss.append(score)
            if printScores==True:
                print(n,score)
        plt.plot(range_n_clusters,ss)
        plt.xticks(range_n_clusters) #so it shows all the ticks    
        
        
def outliers_IQR(df,featurelist,imputestrategy="None",thresh=0.25,printorreturn='print'):
    """
    This is the approach that boxplot uses, which is IQR approach.
    sensitive to null. the more null, the narrower box from both end. boxplot just shrinks, thus number of outliers increases. 
    so it would be sensible to impute the nulls first. we, here, impute them temporarily just in case.
    Args:
        imputestrategy:median, mean, mode, None
        printorreturn:(print,return,both). if print, it prints the results, if return, it returns the list of results as a list of tuple,if both, it prints an returns
    """
    retlist=[]
    for f in featurelist:
        if imputestrategy=='None':
            Q1 = df[f].quantile(thresh)
            Q3 = df[f].quantile(1-thresh)
        else:
            Q1 = df[f].fillna(df[f].agg(imputestrategy)).quantile(thresh)
            Q3 = df[f].fillna(df[f].agg(imputestrategy)).quantile(1-thresh)
        IQR = Q3-Q1
        top=(Q3 + 1.5 * IQR)
        bottom=(Q1 - 1.5 * IQR)
        adet=len(df[(df[f] > top) | (df[f] < bottom)])
        if adet>0:
            if printorreturn=='print':
                print(f"{adet} outliers exists in feature '{f}'")
            elif printorreturn=='return':
                retlist.append((f,adet))
            elif printorreturn=='both':
                retlist.append((f,adet))
                print(f"{adet} outliers exists in feature '{f}'")
            else:
                print("wrong value for printorreturn")
                raise
        
    if printorreturn=='return':
        return retlist        
    
    
def plotRange(df_desc,figsize=(16,3)):
    """
    plots the description data of a dataframe using min, max and median; whether to decide scaling is needed or nor
    """
    plt.figure(figsize=figsize)
    plt.subplot(131)
    df_desc.loc["min"].plot(kind="bar")
    plt.subplot(132)
    df_desc.loc["50%"].plot(kind="bar")
    plt.subplot(133)
    df_desc.loc["max"].plot(kind="bar")
    plt.show();    
    
def getColumnsInLowCardinality(df,i=10,isprint=True):
    """
    prints unique values in a dataframe whose nunique value <= 10.
    Used to find out what columns are in low cardinality(but >1). This shouldn't be used to decide what columns to read as categoric.
    Use this as an input to manual rule finder since there may be too many combinations. Otherwise use 'columnsFromObjToCategory'.
    """     
    try:
        dict_=dict(df.nunique())
        list_=[]
        for k,v in dict_.items():
            if int(v)<=i: #we don't want to see the unique items that are greater than i
                list_.append(k)
                if isprint:
                    print("Unique items in column",k)
                    print(df[k].unique(),end="\n\n")
        print("You may want to consider the numerics with low cardinality as categorical in the analysis")
                
        return list_
    except Exception as e: 
        print(e)    
        
        
def multicountplot(df,i=5,fig=(4,5),r=45, colsize=2,hue=None):  
    """
    countplots for columns whose # of unique value is less than i 
    """
    
    try:        
        dict_=dict(df.nunique())
        target=[k for k,v in dict_.items() if v<=i]
            
        lng=0
        if len(target)<=2:
            print("plot manually due to <=2 target feature")
            return
        if len(target)//colsize==len(target)/colsize:
            lng=len(target)//colsize    
        else:
            lng=len(target)//colsize+1
        
        
        fig, axes= plt.subplots(lng,colsize,figsize=fig)
        k=0    
        for i in range(lng):
            for j in range(colsize):
                if k==len(target):
                    break
                elif target[k]==hue:
                    pass
                else:
                    sns.countplot(x=df[target[k]].fillna("Null"), ax=axes[i,j], data=df, hue=hue)
                    plt.tight_layout()                     
                    axes[i,j].set_xticklabels(axes[i,j].get_xticklabels(), rotation=r,ha='right')
                    k=k+1
    except Exception as e: 
        print(e)
        print("You may want to increase the size of i")        
        
        
def plotNumericsByTarget(df,target,nums=None,layout=None,figsize=None):
    if nums==None:
        nums=df.select_dtypes("number").columns
    grup=df.groupby(target)[nums].mean()
    grup.plot(subplots=True,layout=layout,figsize=figsize,kind="bar",legend=None)
    plt.tight_layout() 
    plt.show();        
    
def GetOnlyOneTriangleInCorr(df,target=None,diagonal=True,heatmap=False,figsize=(12,10)):
    """
    returns the lower part of the correlation matrix. if preffered, heatmap could be plotted.
    args:
        df:dataframe itsel, not the df.corr() result
        diagonal:if False, the values at the diagonal to be made null
    """
    sortedcorr=df.corr().sort_index().sort_index(axis=1)    
    if target is None:
        cols = [col for col in sortedcorr]
    else:
        cols = [col for col in sortedcorr if col != target] + [target]
    sortedcorr = sortedcorr[cols]
    if target is None:
        new_index = sortedcorr.index
    else:
        new_index = [i for i in sortedcorr.index if i != target] + [target]
    sortedcorr=sortedcorr.reindex(new_index)
    for i in range(len(sortedcorr)):
        for c in range(len(sortedcorr.columns)):            
            if diagonal==True:
                if i<c:                
                    sortedcorr.iloc[i,c]=np.nan 
            else:
                if i<=c:                
                    sortedcorr.iloc[i,c]=np.nan 
    
    if target is not None:
        sortedcorr.rename(columns={target: "*"+target+"*"},inplace=True)
        sortedcorr.rename(index={target: "*"+target+"*"},inplace=True)
        
    if heatmap:
        plt.figure(figsize=figsize)
        sns.heatmap(sortedcorr,annot=True)        
    else:
        return sortedcorr     
    
def capOutliers(df,feature,quantile=0.25):
    Q1 = df[feature].quantile(quantile)
    Q3 = df[feature].quantile(1-quantile)
    IQR = Q3-Q1
    top=(Q3 + 1.5 * IQR)
    bottom=(Q1 - 1.5 * IQR)
    df[feature]=np.where(df[feature]>top,top,df[feature])
    df[feature]=np.where(df[feature]<bottom,bottom,df[feature])           


    
#*******************************************************************************    
#*****************************extension ve magics*******************************
#******************************************************************************* 


@register_dataframe_method        
def super_info_(df, dropna=False):
    """
    Returns a dataframe consisting of datatypes, nuniques, #s of nulls head(1), most frequent item and its frequncy,
    where the column names are indices.
    First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    """
    
    dt=pd.DataFrame(df.dtypes, columns=["Type"])
    dn=pd.DataFrame(df.nunique(dropna=dropna), columns=["Nunique"])    
    nonnull=pd.DataFrame(df.isnull().sum(), columns=["#of Missing"])
    firstT=df.head(1).T.rename(columns={0:"First"})
    MostFreqI=pd.DataFrame([df[x].value_counts().head(1).index[0] for x in df.columns], columns=["MostFreqItem"],index=df.columns)
    MostFreqC=pd.DataFrame([df[x].value_counts().head(1).values[0] for x in df.columns], columns=["MostFreqCount"],index=df.columns)
    return pd.concat([dt,dn,nonnull,MostFreqI,MostFreqC,firstT],axis=1)

@register_dataframe_method
def head_and_tail_(df,n=5):    
    """
        Extension method for pandas dataframes. The name is self explanotory.
        First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    """ 
    h=df.head(n)
    t=df.tail(n)
    return pd.concat([h,t],axis=0)

@register_line_magic        
def mgc_suppresswarning(line):
    warnings.filterwarnings("ignore",category=eval(line))
    
def getSepecificColumns_(self,x,incl_or_excl="excl"):
    """
        Extension method for numpy array type. Returns specified columns from 2D array.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
        if incl_or_excl="incl" x must be an array, otherwise an integer
    """
    if incl_or_excl=="incl":
        idx=x #x is array here
    else:
        idx=[i for i in range(self.shape[1]) if i!=x] # x is integer here
    return self[:,idx]

curse(np.ndarray, "getSepecificColumns_", getSepecificColumns_)    


def getItemsContainingLike_(self,what):    
    """
        Extension method for list type. Returns items containing some substrings.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    temp=[]
    for item in set(self):
        if what in str(item):
            temp.append(item)
    return temp    
    
curse(list, "getItemsContainingLike_", getItemsContainingLike_)

def getLongestInnerList_(self):
    """
        Extension method for list type. Returns longest inner list, its length and its index.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """
    longest=sorted(self, key=lambda x:len(x),reverse=True)[0]
    index=self.index(longest)
    return longest,len(longest),index

curse(list, "getLongestInnerList_", getLongestInnerList_)

def valuecounts_(self):
    """
        Extension method for numpy array type. Returns unique counts.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """       
    unique, counts = np.unique(self, return_counts=True)
    return np.asarray((unique, counts)).T

curse(np.ndarray, "valuecounts_", valuecounts_)
