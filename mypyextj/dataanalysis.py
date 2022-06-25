import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
import warnings
import os
from itertools import combinations
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from IPython.display import Markdown, display
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity



# my first initial settings when imported. These are my cup of tea, feel free to change to your preference.
try:
    warnings.simplefilter("always")
    print("all warnings will be shown")
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    pd.set_option('display.max_rows',None)  
    pd.set_option('display.min_rows',None) 
    pd.set_option("io.excel.xlsx.reader", "openpyxl")
    pd.set_option("io.excel.xlsm.reader", "openpyxl")
    pd.set_option("io.excel.xlsb.reader", "openpyxl")
    pd.set_option("io.excel.xlsx.writer", "openpyxl")
    pd.set_option("io.excel.xlsm.writer", "openpyxl")
except:
    pass    

def SuperInfo(df):
    warnings.warn("Warning...SuperInfo is deprecated. Use super_info extension method of dataframe object. this method is located in extensions.py file")

def printUniques(df):
    warnings.warn("Warning...printUniques is deprecated. Use getColumnsInLowCardinality instead")

def printmd(string):
    """
        converts a string as markdown.
    """
    display(Markdown(string))

def pandas_df_to_markdown_table(df):    
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False))) 


#use:df.style.applymap(color_code(1), subset=['col1','col2'])
def color_code(thresh):
    def color_code_by_val(val):
        color = None
        if val <= thresh:
            color = 'red'        
        return 'background-color: %s' % color
    return color_code_by_val  

          
def printValueCount(df,i=10):    
    """
    prints value counts for columns whose # of unique value is less than i 
    """
    try:
        dict_=dict(df.nunique())
        for k,v in dict_.items():
            if int(v)<=i:
                print("Unique items in column",k)
                print(datafr[k].value_counts(dropna=False),end="\n\n")
    except Exception as e: 
        print(e)
                    
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

def columnsFromObjToCategory(df,n=2):
    """
    Used in order to decide the data type(cateogric or object) when reading data. If you just want to see the low cardinality columns, use 'getColumnsInLowCardinality'. Object columns are checked even if you pass the whole dataset.
    n: At what degree you espect the cardinality. Generally accepted rule is that number of unique values must be the half of the total instance at most. If the column is like Name/Surname/Address, it should stay as object, rest should be category.
    """
    dt=pd.DataFrame(df.dtypes, columns=["Type"])    
    dn=pd.DataFrame(df.nunique(), columns=["Nunique(Excl.Nulls)"])
    d=pd.concat([dt,dn],axis=1).sort_index()
    objtocat=d[(d["Nunique(Excl.Nulls)"]<(len(df)/n)) & (d["Type"]=="object")].index.tolist()
    return objtocat

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

def ShowTopN(df,n=5):
    """
    Works for numeric features. Even if you pass categorical features they will be disregarded
    """
    try:
        for c in df.select_dtypes("number").columns:
            print(f"Top {n} in {c}:")
            print(df[c].sort_values(ascending=False).head(n))
            print("---------------------------")
    except Exception as e: 
        print(e)

def sortAndPrintMaxMinNValues(df,columns,n=1,removeNull=True):
    """
       Works for numeric features. Even if you pass categorical features they will be disregarded 
       n: Top N min/max
    """
    #if n=1 returns some unusual values, we can increase n 
    try:   
        for c in df.select_dtypes("number").columns:
            sorted_=df[c].sort_values()        
            if removeNull==True:
                sorted_=sorted_.dropna()
            print((c,sorted_[:n].values,sorted_[-n:].values))        
    except Exception as e: 
        print(e)

def addStdMeanMedian(df):
    warnings.warn("Warning...addStdMeanMedian is deprecated. Use addCoefOfVarianceToDescribe")

def addCoefOfVarianceToDescribe(df):
    """
    First get the describe view of the the df and then adds variance coefficient to the its end.
    """
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

def outliers_std(df,featurelist,n=3,imputestrategy="None",printorreturn='print'): 
    """
    if the std is higher than mean it may go negative at the bottom edge, so you cannot catch bottom outliers
    Args:
        imputestrategy:median, mean, mode, None 
        printorreturn:(print,return,both). if print, it prints the results, if return, it returns the list of results as a list of tuple,if both, it prints an returns
    """
    for f in featurelist:
        if imputestrategy=='None':
            top=df[f].mean()+n*df[f].std()
            bottom=df[f].mean()-n*df[f].std()
        else:
            top=df[f].fillna(df[f].agg(imputestrategy)).mean()+n*df[f].fillna(df[f].agg(imputestrategy)).std()
            bottom=df[f].fillna(df[f].agg(imputestrategy)).mean()-n*df[f].fillna(df[f].agg(imputestrategy)).std()
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
        

def outliers_zs(df,featurelist,thresh_z=3,imputestrategy="None",printorreturn='print'):   
    """
    finds the outliers to the z score.
    Args:
        imputestrategy:median, mean, mode, None 
        printorreturn:(print,return,both). if print, it prints the results, if return, it returns the list of results as a list of tuple,if both, it prints an returns
    """
    for f in featurelist:
        if imputestrategy=='None':
            z= np.abs(stats.zscore(df[f]))
        else:
            z= np.abs(stats.zscore(df[f].fillna(df[f].agg(imputestrategy))))
        
        adet=len(df[np.abs(df[f])>df.iloc[np.where(z>thresh_z)][f].min()])
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

        
def plotHistWithoutOutliers(df,fig=(12,8),thresh=0.25,imputestrategy="median",outliertreat="remove"):
    """this function does not change the dataframe permanently
    args:
        outliertreat: remove or cap
    """
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
       
    
def findNullLikeValues(df,listofvalues=[[-1,-999],["-","na","yok","tanımsız","bilinmiyor","?"]]): 
    """
        df:dataframe,
        listofvalues: turkish words that might mean null. put your own language equivalents.
                      first item in this list are the numeric ones, second one contains strings,
                      default values:[[-1,-999],["na","yok","tanımsız","bilinmiyor","?"]
    """

    if len(listofvalues)<2:
        warnings.warn("The Length of second argument must be 2, first for numerics, second for categoricals")
        return

    t=0
    finallistforobjects=listofvalues[1]+[" "*x for x in range(1,25)]
    values_n=[]
    values_o=[]
    for f in df.select_dtypes("number").columns:
        x=0
        for i in listofvalues[0]:
            r=len(df[df[f]==i])
            if r>0:
                x+=len(df[df[f]==i])
                values_n.append(i)
                t+=1
            
        if x>0:    
            print(f"{x} null-like values,which are '{','.join(set(values_n))}', in {f}")
    for f in df.select_dtypes("object"):
        x=0
        for i in finallistforobjects:
            try: #in case of nulls
                r=len(df[df[f]==i])
                if r>0:                
                    x+=len(df[df[f].str.lower()==i])
                    values_o.append(i)
                    t+=1
            except:
                pass
        if x>0:    
            print(f"{x} null-like values,which are '{','.join(set(values_o))}', in {f}")
    if t==0:
        print("There are no null-like values")
    else:
        print("You may want to replace these values with NaN")
    print("Don't forget to check for 0's manually")    



def plotNumericsBasedOnCategorical(df,cats,nums,fig=(15,15),r=45,aggf='mean',sort=False,hueCol=None):      
    """
    catplot?
    - cast and nums must be array-like.
    - plots will be displayed such that that each numeric feature could be tracked in the rows and categories in the columns
    """
    cols=len(cats)    
    rows=len(nums)
    c=0

    f, axes = plt.subplots(rows,cols,figsize=fig)
    for cat in cats:  
        r=0
        for num in nums:
            ix=axes[r,c] if rows>1 else axes[c]  
            if hueCol is None or hueCol==cat:
                if sort==True:
                    gruplu=df.groupby(cat)[num].agg(aggf).sort_values(ascending=False)
                else:
                    gruplu=df.groupby(cat)[num].agg(aggf)
                  
                sns.barplot(x=gruplu.index, y=gruplu.values,ax=ix)            
            else:
                if sort==True:
                    gruplu=df.groupby([cat,hueCol])[num].agg(aggf).sort_values(ascending=False)
                else:
                    gruplu=df.groupby([cat,hueCol])[num].agg(aggf)
                    
                temp=gruplu.to_frame()
                grupludf=temp.swaplevel(0,1).reset_index()
                sns.barplot(x=cat, y=num,ax=ix, data=grupludf, hue=hueCol)
            
            #plt.xticks(rotation= 45) #isimler uzun olursa horizontalalignment='right' da ekle
            ix.set_xticklabels(ix.get_xticklabels(), rotation=r,ha='right')
            ix.set_title(f"{aggf.upper()} for {num}")
            plt.tight_layout()
            r=r+1
        c=c+1


    
def nullPlot(df):
    warnings.warn("Warning...nullPlot is depreciated. Use missingno's matrix method instead.")
    # sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


def prepareListOfCombinationsForRelationFinder(df,i=5):
    """
    prepares a list of feature combinations of 2-3-4.
    not to be used solo, instead it is called within getListOfRelationsParallel.
    """
    dict_=dict(df.nunique())
    target=[k for k,v in dict_.items() if v<=i]

    if len(target)>50:
        c=3
    elif len(target)>20:
        c=4
    else:
        c=5
    comb=[list(combinations(target,x)) for x in range(2,c)]
    flat_list = [item for sublist in comb for item in sublist]
    return flat_list

def findRelationsAmongFeatures(tpl):
    """
        Must be used with multiprocessing module.
    args
        tpl:tuple consisting of a dataframe and a inner tuple of features of some combinations returning from 'prepareListOfCombinationsForRelationFinder' method. 
        These tuples must be provieded as parallel in a multiprocess-based procedure, which is "getListOfRelationsParallel" below.
    """
    df,item=tpl
    list_=list(item)
    dist=df.drop_duplicates(list_)[list_]
    for i in list_:
        uns = dist[i].unique()
        for u in uns:
            if len(dist[dist[i]==u])==1:
                return (list_,i,uns,u)

def getListOfRelationsParallel(df):

    if __name__ == "__main__":#because of windows-jupyter forking issue, used with "if main"
        cpu=multiprocessing.cpu_count()    
        flat_list=prepareListOfCombinationsForRelationFinder(df)
        tpl=[(df,i) for i in flat_list] 
        with Pool(cpu) as p:
            list_= p.map(findRelationsAmongFeatures, tpl)
        return list_
          
    


def getHighestPairsOfCorrelation(dfcorr,target=None,topN_or_threshold=5):
    if target is None:
        c=dfcorr.abs()
        s=c.unstack()
        sorted_s = s.sort_values(ascending=False)
        final=sorted_s[sorted_s<1]
        if isinstance(topN_or_threshold,int): #top N
            return final[:topN_or_threshold*2:2] #because of the same correlations for left-right and right-left
        else: #threshold
            return final[final>topN_or_threshold][::2]
    else:
        seri=dfcorr[target]
        if isinstance(topN_or_threshold,int): #top N
            return seri[seri.index!=target].sort_values(ascending=False)[:topN_or_threshold]    
        else:
            return seri[seri.index!=target].sort_values(ascending=False)[seri>topN_or_threshold]    

#***************************************COMPARISONS*******************************************
def areContentsOfFeaturesSame(df,features):
    combs=list(combinations(features,2))
    for c in combs:
        if np.all(np.where(df[c[0]] == df[c[1]], True, False)):
            print(f"The content of the features of {c[0]} and {c[1]} are the same")
            
def isFirstAlwaysCompareSecond(df,features,comp,exceptionvalue,sublist=None):
    combs=list(combinations(features,2))
    if sublist is not None:
        combs = [x for x in combs if x[0] in sublist or x[1] in sublist]
    if exceptionvalue is None:
        for c in combs:
            if comp=="equal":
                if np.all(np.where(df[c[0]] == df[c[1]], True, False)):
                    print(f"The content of the features of {c[0]} and {c[1]} are the same")
            elif comp=="lte":
                if np.all(np.where(df[c[0]] <= df[c[1]], True, False)):
                    print(f"The content of the features of {c[0]} is always less than or equal to {c[1]}")
            elif comp=="gte":
                if np.all(np.where(df[c[0]] >= df[c[1]], True, False)):
                    print(f"The content of the features of {c[0]} is always greater than or equal to {c[1]}")
            else:
                print("wrong argument gor comp")
                raise                    
    else:
        for c in combs:
            if comp=="equal":
                if np.all(np.where(np.logical_and(df[c[0]] == df[c[1]],df[c[0]]==exceptionvalue), True, False)):
                    print(f"The content of the features of {c[0]} and {c[1]} are always the same")
            elif comp=="lte":
                if np.all(np.where(np.logical_and(df[c[0]] <= df[c[1]],df[c[0]]==exceptionvalue), True, False)):
                    print(f"The content of the features of {c[0]} is always less than or equal to {c[1]}")
            elif comp=="gte":
                if np.all(np.where(np.logical_and(df[c[0]] >= df[c[1]],df[c[0]]==exceptionvalue), True, False)):
                    print(f"The content of the features of {c[0]} is always greater than or equal to {c[1]}")
            else:
                print("wrong argument gor comp")
                raise              

                
def isComparisonValidOverSpecificRatio(df,features,comp,thresh_ratio=0.99,sublist=None):
    """
        comp=equal,lte,gte. %100 not included, as it is queried above.
    """
    combs=list(combinations(features,2))
    if sublist is not None:
        combs = [x for x in combs if x[0] in sublist or x[1] in sublist]    
    for c in combs:
        if comp=="equal":
            comparison=np.where(df[c[0]] == df[c[1]], True, False)
        elif comp=="lte":
            comparison=np.where(df[c[0]] <= df[c[1]], True, False)
        elif comp=="gte":
            comparison=np.where(df[c[0]] >= df[c[1]], True, False)
        else:
            print("wrong argument gor comp")
            raise
        adet=len(comparison[comparison==True])
        oran=adet/len(df)
        if oran>thresh_ratio and oran<1:
            print(f"The content of the features of {c[0]} and {c[1]} are %{100*oran:.6f} {comp}")  
            
def whatPercentageOfTheTwoFeatureIsComparable(df,features,comp="equal",sublist=None):
    """
        comp=equal,lte,gte
    """
    combs=list(combinations(features,2))
    if sublist is not None:
        combs = [x for x in combs if x[0] in sublist or x[1] in sublist]  
    for c in combs:
        if comp=="equal":
            comparison=np.where(df[c[0]] == df[c[1]], True, False)
        elif comp=="lte":
            comparison=np.where(df[c[0]] <= df[c[1]], True, False)
        elif comp=="gte":
            comparison=np.where(df[c[0]] >= df[c[1]], True, False)
        else:
            print("wrong argument gor comp")
            raise
        adet=len(comparison[comparison==True])
        oran=adet/len(df)
        print(f"The content of the features of {c[0]} and {c[1]} are %{100*oran:.6f} {comp}")  
#***************************************END OF COMPARISONS***************************************
                    
def GetOnlyOneTriangleInCorr(df,target,diagonal=True,heatmap=False):
    """
    returns the lower part of the correlation matrix. if preffered, heatmap could be plotted.
    args:
        diagonal:if False, the values at the diagonal to be made null
    """
    sortedcorr=df.corr().sort_index().sort_index(axis=1)        
    cols = [col for col in sortedcorr if col != target] + [target]
    sortedcorr = sortedcorr[cols]
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
    
    sortedcorr.rename(columns={target: "*"+target+"*"},inplace=True)
    sortedcorr.rename(index={target: "*"+target+"*"},inplace=True)
    if heatmap:
        sns.heatmap(sortedcorr,annot=True)        
    else:
        return sortedcorr       

def plotNumericsByTarget(df,target,nums=None,layout=None,figsize=None):
    if nums==None:
        nums=df.select_dtypes("number").columns
    grup=df.groupby(target)[nums].mean()
    grup.plot(subplots=True,layout=layout,figsize=figsize,kind="bar",legend=None)
    plt.tight_layout() 
    plt.show();

def plotTargetByCats(df, cats, target, subplot_tpl, shrink=0.9,bins=10):
    """
    args:
        cats: categoric features
        tareget: target column
        subplot_tpl: layout of the subplot(r*c)
    """
    r,c=subplot_tpl
    for e,cat in enumerate([x for x in cats if x!=target]):    
        plt.subplot(r,c,e+1)
        (df
        .groupby(cat)[target]
        .value_counts(normalize=True)
        .mul(100)
        .rename('percent')
        .reset_index()
        .pipe((sns.histplot,'data'), x=cat, weights='percent', hue=target, multiple = 'stack', shrink = shrink,bins=bins))
        plt.xticks(rotation= 45,horizontalalignment="right")    
        plt.ylabel("Percentage")

    plt.tight_layout()    
    plt.show();
    
    
def plotPositiveTargetByCats(df, cats, target, subplot_tpl, shrink=0.9,bins=10, pos_label="Yes"):
    """
    args:
        cats: categoric features
        tareget: target column
        subplot_tpl: layout of the subplot(r*c)
    """    
    r,c=subplot_tpl
    for e,cat in enumerate([x for x in cats if x!=target]):
        plt.subplot(r,c,e+1)
        (df[df[target]==pos_label]
        .groupby(target)[cat]
        .value_counts(normalize=True)
        .mul(100)
        .rename('percent')
        .reset_index()
        .pipe((sns.histplot,'data'), x=target, weights='percent', hue=cat, multiple = 'stack', shrink = shrink, bins=bins))
        plt.xticks(rotation= 45,horizontalalignment="right")
        plt.title(cat) 
        
    plt.tight_layout()    
    plt.show();

def plotTargetForNumCatsPairs(df,nums,cats,target,height,aspect):
    for num in nums:
        print(f"Plots for {num},\n----------------------")
        for cat in [x for x in cats if x!=target]:       
            g=sns.catplot(x=target,y=num, data=df, col=cat, kind="bar", height=height, aspect=aspect)        
            plt.show();

def plotCategoricForNumTargetPairs(df,nums,cats,target,height,aspect):
    for num in nums:
        print(f"Plots for {num},\n----------------------")
        for cat in [x for x in cats if x!=target]:
            g=sns.catplot(x=cat,y=num, data=df, col=target, kind="bar", height=height, aspect=aspect)        
            plt.show();


def plotCategoricsByNumericTargetBoxplot(df, numeric_target, categoric_features,col_wrap=2,height=5):
    """
        For regression:

    """
    f = pd.melt(df, id_vars = numeric_target, value_vars = categoric_features)
    g = sns.FacetGrid(f, col = "variable",  col_wrap = col_wrap, sharex = False, sharey = False, height = height)
    plt.axhline(df[numeric_target].mean(), color='r', linestyle='dashed', linewidth=2)
    g.map(sns.boxplot, 'value', numeric_target, palette = 'viridis')


def find_features_for_reading(minmaxdf,sampledf,inversecardinalty=2,droplist=[]):
    """    
    args:
        minmaxdf:dataframe that was read from DB with min and max values of numeric columns
        sampledf:dataframe that was read from DB with a small sample.
        inversecardinalty:boundary for inverse-logic cardinality. Ex:2 means at least 1/2 of the data is unique
        
    Returns returns read_as_float16,read_as_int8,read_as_int16,read_as_int32,read_as_categoric in the given order.
    Use it in sample data and read the whole table from database using the knowledge obtained here, by chunking and converting at every step.        
    """ 
    dict_={}
    #from sampledf    
    read_as_categoric=columnsFromObjToCategory(sampledf.select_dtypes('O'),inversecardinalty)
    read_as_categoric=list(set(read_as_categoric).difference(set(droplist)))
    dict_["cat"]=read_as_categoric
    read_as_date=list(sampledf.select_dtypes(np.datetime64).columns)
    dict_["date"]=read_as_date
    
    #from minmaxdf
    read_as_float16=list(set(minmaxdf[np.logical_and(minmaxdf["Max"]<=127, minmaxdf["Min"]>-128)].index).difference(set(minmaxdf[minmaxdf["Max"].map(lambda x:x==int(x)).values==True].index))) #float8 yok. küçük rakamlarda küsur önemli
    dict_["f16"]=read_as_float16
    read_as_int8=set(minmaxdf[np.logical_and(minmaxdf["Max"]<=127,minmaxdf["Min"]>-128)].index).intersection(set(minmaxdf[minmaxdf["Max"].map(lambda x:x==int(x)).values==True].index)) #0,1 tam binaryler ile -1,1 gibi kısmi binaryler burda, -999,0 gibi kısmi binaryler ise int16
    read_as_int8=list(set(read_as_int8).difference(set(droplist)))
    dict_["i8"]=read_as_int8
    read_as_int16=minmaxdf[np.logical_and(minmaxdf["Max"]<=32767,minmaxdf["Min"]>-32767)].index
    read_as_int16=list(set(read_as_int16).difference(set(read_as_int8)).difference(set(read_as_float16)).difference(set(droplist)))
    dict_["i16"]=read_as_int16
    read_as_int32=minmaxdf[np.logical_and(minmaxdf["Max"]<=2147483647,minmaxdf["Min"]>-2147483647)].index
    read_as_int32=list(set(read_as_int32).difference(set(read_as_int16)).difference(set(read_as_int8)).difference(set(read_as_float16)).difference(set(droplist)))
    dict_["i32"]=read_as_int32
    
    return dict_

def memory_usage(df,istotal):    
    """
        hem pandas hem pyspark df için
        istotal: True ise toplamı, False ise kolon bazında
    """
    if isinstance(df,pyspark.sql.dataframe.DataFrame):
        dfson = df.sample(fraction=0.01).toPandas()
        carpan = 100
    else:
        dfson = df
        carpan=1
    print("printing memoryusage deep")
    if istotal:
        mem=dfson.memory_usage(index=False,deep=True).sum()*carpan/(1024**2)
        print(f"{mem:.2f} MB")
    else:        
        print(dfson.memory_usage(index=False,deep=True)*carpan/(1024**2))
    
    print("\nprinting sys.getsizeof")
    print(f"{sys.getsizeof(dfson)/(1024**2)*carpan:.2f} MB")
    del dfson

def discretize_numerics(df,numerics,droporjinalcolumns=False):
    """
    args:
        numerics:numeric type features
    returns additional features with the suffix '_sign'
    """
    for c in numerics:
        df[c+"_sign"]=np.sign(df[c])   
    if droporjinalcolumns:
        df.drop(numerics,axis=1,inplace=True)
        
def discretize_dates(df,dates,droporjinalcolumns=False):
    """
    args:
        numerics:numeric type features
    returns additional features with the suffix '_isnull'
    """    
    for t in dates:
        df[t+"_isnull"]=np.where(np.isnan(df[t]),1,0)
    if droporjinalcolumns:
        df.drop(dates,axis=1,inplace=True)

