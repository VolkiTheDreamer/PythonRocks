import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,roc_auc_score,roc_curve
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.utils import column_or_1d
import warnings
from sklearn.pipeline import Pipeline 
import os, sys, site
import functools
import time
import itertools

##docstring yapılcak


#*******************************************************************************
#***************************************General*********************************
#*******************************************************************************

def doInitialSettings(figsize=(5,3)):
    warnings.simplefilter("always")
    multioutput()
    plt.rcParams["figure.figsize"] = figsize  
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    pd.set_option('display.max_rows',20)  
    pd.set_option("io.excel.xlsx.reader", "openpyxl")
    pd.set_option("io.excel.xlsm.reader", "openpyxl")
    pd.set_option("io.excel.xlsb.reader", "openpyxl")
    pd.set_option("io.excel.xlsx.writer", "openpyxl")
    pd.set_option("io.excel.xlsm.writer", "openpyxl")


def multioutput(type="all"):
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = type

    
def scriptforReload():
    print("""
    %load_ext autoreload
    %autoreload 2""")
   
    
def scriptforTraintest():
    print("X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)")
    
def scriptForCitation():
    print("""<p style="font-size:smaller;text-align:center">Görsel <a href="url">bu sayfadan</a> alınmıştır</p>""")
    
def pythonSomeInfo():
    print("system packages folder:",sys.prefix, end="\n\n")
    print("pip install folder:",site.getsitepackages(), end="\n\n")    
    print("python version:", sys.version, end="\n\n")
    print("executables location:",sys.executable, end="\n\n")
    print("pip version:", os.popen('pip version').read(), end="\n\n")
    pathes= sys.path
    print("Python pathes")
    for p in pathes:
        print(p)

def timeElapse(func):
    """
        usage:
        @timeElapse
        def somefunc():
            ...
            ...
            
        somefunc()
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        start=time.time()
        value=func(*args,**kwargs)
        func()
        finito=time.time()
        print("Time elapsed:{}".format(finito-start))
        return value
    return wrapper    
#*******************************************************************************
#******************************EDA and Analysis*********************************
#*******************************************************************************

def printUniques(datafr,i=10):    
    dict_=dict(datafr.nunique())
    for k,v in dict_.items():
        if int(v)<=i: #we don't want to see the unique items that are greater than i
            print("Unique items in column",k)
            print(datafr[k].unique(),end="\n\n")
            
def printValueCount(datafr,i=10):    
    #prints value counts for columns whose # of unique value is less than i 
    dict_=dict(datafr.nunique())
    for k,v in dict_.items():
        if int(v)<=i:
            print("Unique items in column",k)
            print(datafr[k].value_counts(dropna=False),end="\n\n")
            
def getColumnsInLowCardinality(df,i):
    dict_=dict(df.nunique())
    list_=[]
    for k,v in dict_.items():
        if int(v)<=i:
            list_.append(k)
            
    return list_


def multicountplot(datafr,i=5,fig=(4,5),r=45, colsize=1):  
    """countplots for columns whose # of unique value is less than i """
    
    dict_=dict(datafr.nunique())
    target=[k for k,v in dict_.items() if v<=i]
        
    lng=0
    if len(target)//colsize==len(target)/colsize:
        lng=len(target)//colsize    
    else:
        lng=len(target)//colsize+1
    
    
    fig, axes= plt.subplots(lng,colsize,figsize=fig)
    k=0
    for i in range(lng):
        for j in range(colsize):
            sns.countplot(datafr[target[k]].fillna("Null"), label="Count",ax=axes[i,j])
            plt.tight_layout() 
            axes[i,j].set_xticklabels(axes[i,j].get_xticklabels(), rotation=r,ha='right')
            k=k+1
                    
def ShowTopN(df,n=5):
    for d in df.select_dtypes("number").columns:
        print(f"Top {n} in {d}:")
        print(df[d].sort_values(ascending=False).head(n))
        print("---------------------------")

def sortAndPrintMaxMinNValues(df,columns,n=1,removeNull=True):
    #if n=1 returns some unusual values, we can increase n    
    for c in columns:
        sorted_=df[c].sort_values()        
        if removeNull==True:
            sorted_=sorted_.dropna()
        print((c,sorted_[:n].values,sorted_[-n:].values))        
            
def addStdMeanMedian(df):
    df=df.describe().T
    df["mean/median"]=df["mean"]/df["50%"]
    df["std/mean"]=df["std"]/df["mean"]
    return df

def outlierinfo(df,featurelist,imputestrategy="None",thresh=0.25):
    """
    Gives Q1,Q3,IQR, outlier beginning points, mean in the boxplot, total mean.
    Args:
        imputestrategy:median, mean, mode, None
    """
    for f in featurelist:
        if imputestrategy=='None':
            Q1 = df[f].quantile(thresh)
            Q3 = df[f].quantile(1-thresh)
            IQR = Q3-Q1
            top=(Q3 + 1.5 * IQR)
            bottom=(Q1 - 1.5 * IQR)            
            mbox=df[(df[f] > top) | (df[f] < bottom)][f].mean()
            m=df[f].mean()
            outliers=len(df[(df[f]>top) | (df[f]<bottom)])
        else:
            temp=df[f].fillna(df[f].agg(imputestrategy))
            Q1 = temp.quantile(thresh)
            Q3 = temp.quantile(1-thresh)
            IQR = Q3-Q1
            top=(Q3 + 1.5 * IQR)
            bottom=(Q1 - 1.5 * IQR)            
            mbox=temp[(temp > top) | (temp < bottom)].mean()
            m=temp.mean()            
            outliers=len(temp[(temp >top) | (temp<bottom)])

        print(f"{f}, Min:{df[f].min()}, Max:{df[f].max()}, Q1:{Q1:9.2f}, Q3:{Q3:9.2f}, IQR:{IQR:9.2f}, Q3+1,5*IQR:{top:9.2f}, Q1-1,5*IQR:{bottom:9.2f}, Mean within the box:{mbox:9.2f}, Total Mean:{m:9.2f}, Outliers:{outliers}",end="\n\n")
        

def outliers1(df,featurelist,imputestrategy="None",thresh=0.25):
    """
    This is the approach that boxplot uses.
    sensitive to null. the more null, the narrower box from both end. boxplot just shrinks, thus number of outliers increases. 
    so it would be sensible to impute the nulls first. we, here, impute them temporarily just in case.
    Args:
        imputestrategy:median, mean, mode, None
    """
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
        print("{} outliers exists in feature '{}'".format(len(df[(df[f] > top) | (df[f] < bottom)]),f))

def outliers2(df,featurelist,n=3,imputestrategy="None"): 
    """
    if the std is higher than mean it may go negative at the bottom edge, so you cannot catch bottom outliers
    Args:
        imputestrategy:median, mean, mode, None    
    """
    for f in featurelist:
        if imputestrategy=='None':
            top=df[f].mean()+n*df[f].std()
            bottom=df[f].mean()-n*df[f].std()
        else:
            top=df[f].fillna(df[f].agg(imputestrategy)).mean()+n*df[f].fillna(df[f].agg(imputestrategy)).std()
            bottom=df[f].fillna(df[f].agg(imputestrategy)).mean()-n*df[f].fillna(df[f].agg(imputestrategy)).std()
        print("{} outliers exists in feature '{}'".format(len(df[(df[f]>top) | (df[f]<bottom)]),f))
        

def outliers3(df,featurelist,thresh_z=3,imputestrategy="None"):   
    """
    finds the outliers to the z score.
    Args:
        imputestrategy:median, mean, mode, None    
    """
    for f in featurelist:
        if imputestrategy=='None':
            z= np.abs(stats.zscore(df[f]))
        else:
            z= np.abs(stats.zscore(df[f].fillna(df[f].agg(imputestrategy))))
        print("{} outliers exists in feature '{}'".format(len(df[np.abs(df[f])>df.iloc[np.where(z>thresh_z)][f].min()]),f))

        
def plotHistWithoutOutliers(df,fig=(12,8),thresh=0.25,imputestrategy="median",outliertreat="remove"):
    """this function does not change the dataframe permanently"""
    df=df.select_dtypes("number")
    
    col=4
    row=int(len(df.columns)/col)+1
    _, axes = plt.subplots(row,col,figsize=fig)
    delete=row*col-len(df.columns)
    for d in range(delete):
        plt.delaxes(axes[row-1,col-d-1])
        
    plt.suptitle("Histograms without outliers")
    r=0;c=0;fc=0;
    for f in sorted(df.columns):
        Q1 = df[f].fillna(df[f].agg(imputestrategy)).quantile(thresh)
        Q3 = df[f].fillna(df[f].agg(imputestrategy)).quantile(1-thresh)
        IQR = Q3-Q1
        t1=(Q3 + 1.5 * IQR)
        t2=(Q1 - 1.5 * IQR)
        cond=((df[f] > t1) | (df[f] < t2))
        
        r=int(fc/4)
        c=fc % 4
        if outliertreat=="remove":                
            df[~cond][f].hist(ax=axes[r,c])
        elif outliertreat=="cap":
            s=df[f].copy()
            s.where(s>t2,t2,inplace=True)
            s.where(s<t1,t1,inplace=True)
            s.hist(ax=axes[r,c])
        else:
            print("wrong value for outliertreat")
            raise
        #axes[r,c].set_xticklabels(axes[r,c].get_xticklabels(), rotation=r,ha='right')
        axes[r,c].set_title(f)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fc=fc+1
       
    
        
def numpyValuecounts(dizi):
    unique, counts = np.unique(dizi, return_counts=True)
    return np.asarray((unique, counts)).T


def findNullLikeValues(df,listofvalues=[[-1,-999],["na","yok","tanımsız","bilinmiyor","?"]]): 
    """
        df:dataframe,
        listofvalues: turkish words that might mean null. put your own language equivalents.
                      first item in this list are the numeric ones, second one contains strings,
                      default values:[[-1,-999],["na","yok","tanımsız","bilinmiyor","?"]
    """
    for f in df.select_dtypes("number").columns:
        x=0
        for i in listofvalues[0]: #first is 
            x=x+len(df[df[f]==i])
        print("{} null-like values in {}".format(x,f))
    for f in df.select_dtypes("object"):
        x=0
        for i in listofvalues[1]:
            x=x+len(df[df[f].str.lower()==i])
        print("{} null-like values in {}".format(x,f))
        

def parse_col_json(column, key):
    """
    Args:
        column: string
            name of the column to be processed.
        key: string
            name of the dictionary key which needs to be extracted
    """
    for index,i in zip(movies_df.index,movies_df[column].apply(json.loads)):
        list1=[]
        for j in range(len(i)):
            list1.append((i[j][key]))# the key 'name' contains the name of the genre
        movies_df.loc[index,column]=str(list1)

def plotNumericsBasedOnCategorical(df,cats,nums,fig=(15,15),r=45,aggf='mean',sort=False,hueCol=None):        
    cols=len(cats)    
    rows=len(nums)
    c=0

    f, axes = plt.subplots(rows,cols,figsize=fig)
    for cat in cats:  
        r=0
        for num in nums:
            if hueCol is None or hueCol==cat:
                if sort==True:
                    gruplu=df.groupby(cat)[num].agg(aggf).sort_values(ascending=False)
                else:
                    gruplu=df.groupby(cat)[num].agg(aggf)
                    
                sns.barplot(x=gruplu.index, y=gruplu.values,ax=axes[r,c])            
            else:
                if sort==True:
                    gruplu=df.groupby([cat,hueCol])[num].agg(aggf).sort_values(ascending=False)
                else:
                    gruplu=df.groupby([cat,hueCol])[num].agg(aggf)
                    
                temp=gruplu.to_frame()
                grupludf=temp.swaplevel(0,1).reset_index()
                sns.barplot(x=cat, y=num,ax=axes[r,c], data=grupludf, hue=hueCol)
            
            #plt.xticks(rotation= 45) #isimler uzun olursa horizontalalignment='right' da ekle
            axes[r,c].set_xticklabels(axes[r,c].get_xticklabels(), rotation=r,ha='right')
            axes[r,c].set_title(f"{aggf.upper()} for {num}")
            plt.tight_layout()
            r=r+1
        c=c+1


def countifwithConditon(df,feature,condition):
    print(df[df[feature].isin(df[condition][feature])].groupby(feature).size().value_counts())
    
def nullPlot(df):
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    

def checkCardinality(df):
    pass

def checkRarity(df):
    pass


#*******************************************************************************
#**************************Machine Learning/Data Science************************
#*******************************************************************************
def printScores(y_test,y_pred,*, alg_type='c'):
    """
    
    Args:
    alg_type: c for classfication, r for regressin
    """
    if alg_type=='c':
        print("Accuracy:",accuracy_score(y_test,y_pred))
        print("Recall:",recall_score(y_test,y_pred))
        print("Precision:",precision_score(y_test,y_pred))
        print("F1:",f1_score(y_test,y_pred))
    else:
        print("RMSE:",mean_squared_error(y_test,y_pred))
        print("MAE:",mean_absolute_error(y_test,y_pred))
        print("r2:",r2_score(y_test,y_pred))

def draw_siluet(range_n_clusters,data,isbasic=True,printScores=True):
    if isbasic==False:
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
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(data)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
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
    else:
        ss = []
        for n in range_n_clusters:
            kmeans = KMeans(n_clusters=n)
            kmeans.fit_transform(data)
            labels = kmeans.labels_
            score = silhouette_score(data, labels)
            ss.append(score)
            if printScores==True:
                print(n,score)
        plt.plot(range_n_clusters,ss)

        
def draw_elbow(ks,data):
    wcss = []
    for i in ks:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) #k-means++ ensures that you get don’t fall into the random initialization trap.???????
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(ks, wcss)
    plt.title('Elbow Method')
    plt.xlabel('# of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
#PCA biplot    
def biplot(score,coeff,y,variance,labels=None):
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

    

def PCAChart(X_pca,alpha=0.2):
    n=X_pca.shape[1] #second dimension is the number of colums which is the number of components
    if n==2:
        plt.scatter(X_pca[:,0], X_pca[:,1],alpha=alpha);
    elif n==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.scatter(ax,xs=X_pca[:,0], ys=X_pca[:,1],zs=X_pca[:,2],alpha=alpha)
    else:
        print("n should be either 2 or 3")
        
def getfullitemsforOHE(wholedf,featlist,sort=True):
    """
    wholedf should be the dataframe including both train and test set.
    """
    def sortornot(X):
        if sort==False:
            return X
        else:
            return sorted(X)
       
    fulllist=[]
    for feat in featlist:
        fulllist.append(sortornot(wholedf[feat].unique()))
    return fulllist

def getfeaturenames(ct,dataframe):
    final_features=[]
    
    for trs in ct.transformers_:
        trName=trs[0]
        trClass=trs[1]
        features=trs[2]
        if isinstance(trClass,Pipeline):   
            n,tr=zip(*trClass.steps)
            for t in tr: #t is a transformator object, tr is the list of all transoformators in the pipeline                
                if isinstance(t,OneHotEncoder):
                    for f in t.get_feature_names(features):
                        final_features.append("OHE_"+f) 
                    break
            else: #if not found onehotencoder, add the features directly
                for f in features:
                    final_features.append(f)                
        elif isinstance(trClass,OneHotEncoder): #?type(trClass)==OneHotEncoder:
            for f in trClass.get_feature_names(features):
                final_features.append("OHE_"+f) 
        else:
            #remainders
            if trName=="remainder":
                for i in features:
                    final_features.append(list(dataframe.columns)[i])
            #all the others
            else:
                for f in features:
                    final_features.append(f)                

    return final_features 

def featureImportanceEncoded(importance,feature_names,figsize=(8,6)):
    plt.figure(figsize=figsize)
    dfimp=pd.DataFrame(importance.reshape(-1,1).T,columns=feature_names).T
    dfimp.index.name="Encoded"
    dfimp.rename(columns={0: "Importance"},inplace=True)
    dfimp.reset_index(inplace=True)
    dfimp["Feature"]=dfimp["Encoded"].apply(lambda x:x[4:].split('_')[0] if "OHE" in x else x)
    dfimp.groupby(by='Feature')["Importance"].sum().sort_values().plot(kind='barh');
    
    
def compareClassifiers(gs,tableorplot='plot',figsize=(10,5)):
    cvres = gs.cv_results_
    cv_results = pd.DataFrame(cvres)
    cv_results['param_clf']=cv_results['param_clf'].apply(lambda x:str(x).split('(')[0])
    cols={"mean_test_score":"MAX of mean_test_score","mean_fit_time":"MIN of mean_fit_time"}
    summary=cv_results.groupby(by='param_clf').agg({"mean_test_score":"max", "mean_fit_time":"min"}).rename(columns=cols)
    summary.sort_values(by='MAX of mean_test_score', ascending=False,inplace=True)
    
    
    if tableorplot=='table':
        return summary
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        color = 'tab:red'
        ax1.set_xticklabels('Classifiers', rotation=45,ha='right')
        
        ax1.set_ylabel('MAX of mean_test_score', color=color)
        ax1.bar(summary.index, summary['MAX of mean_test_score'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        

        ax2 = ax1.twinx() 

        color = 'tab:blue'
        ax2.set_ylabel('MIN of mean_fit_time', color=color) 
        ax2.plot(summary.index, summary['MIN of mean_fit_time'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.show()    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')        