from forbiddenfruit import curse
from pandas_flavor import register_dataframe_method,register_series_method
import numpy as np
import pandas as pd

# I chosed to create the pandas extension with pandas_flavor. We can use forbidden_fruit, though. But since its syntax is simpler i prefer the former.

def getLongestInnerList(self):
    """
        Extension method for list type. Returns longest inner list, its length and its index.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """
    longest=sorted(self, key=lambda x:len(x),reverse=True)[0]
    index=self.index(longest)
    return longest,len(longest),index

curse(list, "getLongestInnerList", getLongestInnerList)

def getFirstItemFromDictionary(self):
    """
        Extension method for dict type. Gets the first item from a dictionary.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    return next(iter(self)),next(iter(self.values()))   
curse(dict, "getFirstItemFromDictionary", getFirstItemFromDictionary)


def removeItemsFromList(self,list2,inplace=True):    
    """
        Extension method for list type. Removes items from list2 from list1.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    if inplace:
        for x in set(list2):
            self.remove(x)
        return self
    else:
        temp=self.copy()
        for x in set(list2):
            temp.remove(x)
        return temp
    
curse(list, "removeItemsFromList", removeItemsFromList)


def containsLike(self,what):    
    """
        Extension method for list type. Returns items containing some substrings.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    for item in set(self):
        if what in str(item):
            return True
    else:
        return False    
    
curse(list, "containsLike", containsLike)


def valuecounts(self):
    """
        Extension method for numpy array type. Returns items containing some substrings.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """       
    unique, counts = np.unique(self, return_counts=True)
    return np.asarray((unique, counts)).T

curse(np.array, "valuecounts", valuecounts)


#********************************
#Extensions for Pandas DataFrames
#********************************

@register_dataframe_method
def search(df,lookup_value):
    """
        Extension method for pandas dataframes. With this, you can search a value withing whole dataframe just like you do in Excel.
        First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/        
    """
    for c in df.columns:
        if lookup_value in df[c].astype('str').unique():
            print(f"Found in feature {c}")
            break
            
            
@register_dataframe_method
def nullColumns(df):
    """
        Extension method for pandas dataframes. With this, you return the null-containing column names..
        First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    """    
    return df.columns[df.isna().any()].tolist()


@register_dataframe_method
def head_and_tail(df,n=5):    
    """
        Extension method for pandas dataframes. The name is self explanotory.
        First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    """ 
    h=df.head(n)
    t=df.tail(n)
    return pd.concat([h,t],axis=1)

       
@register_dataframe_method        
def SuperInfo(df, dropna=False):
    """
    Returns a dataframe consisting of datatypes, nuniques, #s of nulls head(1), most frequent item and its frequncy,
    where the column names are indices.
    """
    
    dt=pd.DataFrame(df.dtypes, columns=["Type"])
    dn=pd.DataFrame(df.nunique(dropna=dropna)), columns=["Nunique"])    
    nonnull=pd.DataFrame(df.isnull().sum(), columns=["#of Missing"])
    firstT=df.head(1).T.rename(columns={0:"First"})
    MostFreqI=pd.DataFrame([df[x].value_counts().head(1).index[0] for x in df.columns], columns=["MostFreqItem"],index=df.columns)
    MostFreqC=pd.DataFrame([df[x].value_counts().head(1).values[0] for x in df.columns], columns=["MostFreqCount"],index=df.columns)
    return pd.concat([dt,dn,nonnull,MostFreqI,MostFreqC,firstT],axis=1)
        