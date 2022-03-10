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


def printmd(string):
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

def doInitialSettings(figsize=(5,3)):
    try:
        warnings.simplefilter("always")
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = 'all'
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
    except:
        pass


def printUniques(datafr,i=10):   
    """
    prints unique values in a dataframe whose nunique value <= 10 
    """ 
    try:   
        dict_=dict(datafr.nunique())
        for k,v in dict_.items():
            if int(v)<=i: #we don't want to see the unique items that are greater than i
                print("Unique items in column",k)
                print(datafr[k].unique(),end="\n\n")
        print("You may want to convert the numerics with low cardinality to categorical")
    except Exception as e: 
        print(e)

            
def printValueCount(datafr,i=10):    
    """
    prints value counts for columns whose # of unique value is less than i 
    """
    try:
        dict_=dict(datafr.nunique())
        for k,v in dict_.items():
            if int(v)<=i:
                print("Unique items in column",k)
                print(datafr[k].value_counts(dropna=False),end="\n\n")
    except Exception as e: 
        print(e)
                    
def getColumnsInLowCardinality(df,i=10):
    #buna gerek var mı ? printUniques fakrı ne?
    try:
        dict_=dict(df.nunique())
        list_=[]
        for k,v in dict_.items():
            if int(v)<=i:
                list_.append(k)
                
        return list_
    except Exception as e: 
        print(e)

def multicountplot(datafr,i=5,fig=(4,5),r=45, colsize=2,hue=None):  
    """countplots for columns whose # of unique value is less than i """
    
    try:        
        dict_=dict(datafr.nunique())
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
                    sns.countplot(x=datafr[target[k]].fillna("Null"), ax=axes[i,j], data=datafr, hue=hue)
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
        for d in df.select_dtypes("number").columns:
            print(f"Top {n} in {d}:")
            print(df[d].sort_values(ascending=False).head(n))
            print("---------------------------")
    except Exception as e: 
        print(e)

def sortAndPrintMaxMinNValues(df,columns,n=1,removeNull=True):
    #if n=1 returns some unusual values, we can increase n 
    try:   
        for c in columns:
            sorted_=df[c].sort_values()        
            if removeNull==True:
                sorted_=sorted_.dropna()
            print((c,sorted_[:n].values,sorted_[-n:].values))        
    except Exception as e: 
        print(e)

def addStdMeanMedian(df):
    warnings.warn("Warning...addStdMeanMedian is depreciated. Use addCoefOfVarianceToDescribe")

def addCoefOfVarianceToDescribe(df):
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
    t=0
    for f in df.select_dtypes("number").columns:
        x=0
        for i in listofvalues[0]:
            x+=len(df[df[f]==i])
            t+=1
            
        if x>0:    
            print("{} null-like values in {}".format(x,f))
    for f in df.select_dtypes("object"):
        x=0
        for i in listofvalues[1]:
            try: #in case of nulls
                x+=len(df[df[f].str.lower()==i])
                t+=1
            except:
                pass
        if x>0:    
            print("{} null-like values in {}".format(x,f))
    if t==0:
        print("There are no null-like values")

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
    """
    BUNA SANKİ GEREK YOK GİBİ; CATPLOT YAPIYOR BU İŞİ, HW1'den kontrol et
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


def countifwithConditon(df,feature,condition):
    print(df[df[feature].isin(df[condition][feature])].groupby(feature).size().value_counts())
    
def nullPlot(df):
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
        

def SuperInfo(df, dropna=False):
    """
    Returns a dataframe consisting of datatypes, nuniques, #s of nulls head(1), most frequent item and its frequncy,
    where the column names are indices.
    """
    
    dt=pd.DataFrame(df.dtypes, columns=["Type"])
    dn=pd.DataFrame(df.nunique(), columns=["Nunique"])
    nonnull=pd.DataFrame(df.isnull().sum(), columns=["#of Missing"])
    firstT=df.head(1).T.rename(columns={0:"First"})
    MostFreqI=pd.DataFrame([df[x].value_counts().head(1).index[0] for x in df.columns], columns=["MostFreqItem"],index=df.columns)
    MostFreqC=pd.DataFrame([df[x].value_counts().head(1).values[0] for x in df.columns], columns=["MostFreqCount"],index=df.columns)
    return pd.concat([dt,dn,nonnull,MostFreqI,MostFreqC,firstT],axis=1)


def prepareListOfCombinationsForRelationFinder(df,i=5):
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
        tpl:tuple consisting of a dataframe and a inner tuple of features of some combinations returning from 'prepareListOfCombinationsForRelationFinder' method. These tuples must be provieded as parallel in a multiprocess-based procedure. 
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
    if __name__ == "__main__":#windows-jupyter olayı nedeniyle if main
        cpu=multiprocessing.cpu_count()    
        flat_list=prepareListOfCombinationsForRelationFinder(df)
        tpl=[(df,i) for i in flat_list] 
        with Pool(cpu) as p:
            list_= p.map(findRelationsAmongFeatures, tqdm(tpl))
        return list_
            


    
def pandas_df_to_markdown_table(df):    
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))    
    
def topNValExcluded(serie, n):
    return serie[~serie.isin(serie.nlargest(10).values)]

def getHighestPairsOfCorrelation(dfcorr,top=5):
    c=dfcorr.abs()
    s=c.unstack()
    sorted_s = s.sort_values(ascending=False)
    final=sorted_s[sorted_s<1]
    return final[:top*2:2] #because of the same correlations for left-right and right-left

def areContentsOfFeaturesSame(df,features):
    combs=list(combinations(features,2))
    for c in combs:
        if np.all(np.where(df[c[0]] == df[c[1]], True, False)):
            print(f"The content of the features of {c[0]} and {c[1]} are the same")
            
def calculateManhattanSimilarity(df,features,typeofsim="classic",threshold=0.01):            
    combs=list(combinations(features,2))
    dict_={}
    for c in combs:
        manhattan=distance.cityblock(df[c[0]].fillna(0).values,df[c[1]].fillna(0).values)
        if typeofsim=="classic":
            manhattansim=1/(1+manhattan)    
        elif typeofsim=="divbymean":
            manhattansim=manhattan/np.mean(df[c[0]].fillna(0).values)    
        else:
            print("wrong value for typeofsim")
            raise
        if manhattansim>threshold:
            dict_[(c[0],c[1])]=(int(manhattan),manhattansim)
    newdict={k: v for k, v in sorted(dict_.items(), key=lambda item: item[1])}
    for k,v in newdict.items():
        print(k,v)
        
def calculateCosineSimilarity(df,features,threshold=0.9):            
    combs=list(combinations(features,2))
    dict_={}
    for c in combs:        
        cossim=cosine_similarity(df[c[0]].fillna(0).values.reshape(1,-1),df[c[1]].fillna(0).values.reshape(1,-1))        
        if cossim>threshold:
            dict_[(c[0],c[1])]=(cossim)
    newdict={k: v for k, v in sorted(dict_.items(), key=lambda item: item[1])}
    for k,v in newdict.items():
        print(k,v)
        
def getCartesian(*listeler):   
    kartezyen=itertools.product(*listeler)
    os.chdir(r"c:\users\N35516\desktop")
    csvfile = "kartezyen.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(kartezyen)
                    
def GetOnlyOneTriangleInCorr(df,target,whichpart="lower",diagonal=True,heatmap=False):
    sortedcorr=df.corr().sort_index().sort_index(axis=1)        
    cols = [col for col in sortedcorr if col != target] + [target]
    sortedcorr = sortedcorr[cols]
    new_index = [i for i in sortedcorr.index if i != target] + [target]
    sortedcorr=sortedcorr.reindex(new_index)
    for i in range(len(sortedcorr)):
        for c in range(len(sortedcorr.columns)):
            if whichpart=="lower":
                if diagonal==True:
                    if i<c:                
                        sortedcorr.iloc[i,c]=np.nan 
                else:
                    if i<=c:                
                        sortedcorr.iloc[i,c]=np.nan 

            else:
                if diagonal==True:
                    if i>c:                
                        sortedcorr.iloc[i,c]=np.nan 
                else:
                    if i>=c:                
                        sortedcorr.iloc[i,c]=np.nan 
    
    sortedcorr.rename(columns={target: "*"+target+"*"},inplace=True)
    sortedcorr.rename(index={target: "*"+target+"*"},inplace=True)
    if heatmap:
        sns.heatmap(sortedcorr,annot=True)        
    else:
        return sortedcorr       

def plotTargetByCats(df, cats, target, subplot_tpl=(r,c),shrink=0.9,bins=10)
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