from logging import warning
from forbiddenfruit import curse
from pandas_flavor import register_dataframe_method,register_series_method
import numpy as np
import pandas as pd
import warnings

# I chose to create the pandas extension with pandas_flavor. We can use forbidden_fruit, though. 
# But since its syntax is simpler i prefer the former.

#*************************************string**************************************
def contains_(self,lookup):
    """
    same as "xx in a-string"
    """
    return lookup in self

curse(str, "contains_", contains_)    

#**************************************list*******************************************
def intersect_(self,list2,inplace=True):
    """
        Extension method for list type. It works like sets' counterpart.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    if inplace:
        return list(set(self).intersection(set(list2)))
    else:
        temp=self.copy()
        return list(set(temp).intersection(set(list2)))
curse(list, "intersect", intersect)

def difference_(self,list2,inplace=True):
    """
        Extension method for list type. It works like sets' counterpart
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """ 
    if inplace:
        return list(set(self).difference(set(list2)))
    else:
        temp=self.copy()
        return list(set(temp).difference(set(list2)))
curse(list, "difference", difference)


def union_(self,list2,inplace=True):
    """
        Extension method for list type. It works like sets' counterpart.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """        
    if inplace:
        return list(set(self).union(set(list2)))
    else:
        temp=self.copy()
        return list(set(temp).union(set(list2)))
curse(list, "union", union)


def getLongestInnerList_(self):
    """
        Extension method for list type. Returns longest inner list, its length and its index.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """
    longest=sorted(self, key=lambda x:len(x),reverse=True)[0]
    index=self.index(longest)
    return longest,len(longest),index

curse(list, "getLongestInnerList_", getLongestInnerList_)

def removeItemsFromList_(self,list2,inplace=True):    
    """
        Extension method for list type. Deprecated. use removeItems_.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    warning.warn("Deprecated. use removeItems_.") 
    
curse(list, "removeItemsFromList_", removeItemsFromList_)

def removeItems_(self,list2,inplace=True):    
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
    
curse(list, "removeItems_", removeItems_)


def containsLike_(self,what):    
    """
        Extension method for list type. Returns True if at least 1 item contains some substrings.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    for item in set(self):
        if what in str(item):
            return True
    else:
        return False    
    
curse(list, "containsLike_", containsLike_)

def getItemsContainingLike_(self,what):    
    """
        Extension method for list type. Returns items containing some substrings.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    for item in set(self):
        if what in str(item):
            return True
    else:
        return False    
    
curse(list, "getItemsContainingLike", getItemsContainingLike)


def getMultipleItems_(self,indices):    
    """
        Extension method for list type. Returns items with specified indices.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    return [self[i] for i in indices]
    
curse(list, "getMultipleItems_", getMultipleItems_)

def getIndicesOfItem_(self,item):    
    """
        Extension method for list type. Returns (multiple) indices of an item.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    return [e for e,v in enumerate(self) if v==item]
    
curse(list, "getIndicesOfItem_", getIndicesOfItem_)

#**************************************dict*******************************************

def getFirstItemFromDictionary_(self):
    """
        Extension method for dict type. Gets the first item from a dictionary.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """    
    return next(iter(self)),next(iter(self.values()))   
curse(dict, "getFirstItemFromDictionary_", getFirstItemFromDictionary_)

def sortbyValue_(self,inplace=True):    
    """
        Extension method for dicitonary type. Sorts the dictionary by values.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """
    if inplace:
        self = sorted(self.items(), key=lambda x: x[1], reverse=True)    
        return self
    else:
        return sorted(self.items(), key=lambda x: x[1], reverse=True)    
    
curse(dict, "sortbyValue_", sortbyValue_)



#**************************************numpy*******************************************

def valuecounts_(self):
    """
        Extension method for numpy array type. Returns unique counts.
        First, forbiddenfruit must be installed via https://pypi.org/project/forbiddenfruit/
    """       
    unique, counts = np.unique(self, return_counts=True)
    return np.asarray((unique, counts)).T

curse(np.ndarray, "valuecounts_", valuecounts_)






#********************************
#Extensions for Pandas DataFrames/Series
#********************************

@register_dataframe_method
def search_(df,lookup_value):
    """
        Extension method for pandas dataframes. With this, you can search a value withing whole dataframe just like you do in Excel.
        First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/        
    """
    for c in df.columns:
        if lookup_value in df[c].astype('str').unique():
            print(f"Found in feature {c}")
            break
            
            
@register_dataframe_method
def nullColumns_(df):
    """
        Extension method for pandas dataframes. With this, you return the null-containing column names..
        First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    """    
    return df.columns[df.isna().any()].tolist()


@register_dataframe_method
def head_and_tail_(df,n=5):    
    """
        Extension method for pandas dataframes. The name is self explanotory.
        First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    """ 
    h=df.head(n)
    t=df.tail(n)
    return pd.concat([h,t],axis=1)

       
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
def argwhere_(df,column,value):
    """
    Returns the index of the value in the given column.    
    First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    """    
    return df[df[column]==value].index[0]

@register_dataframe_method        
def getRowOnAggregation_(df,col,agg_):
    """
    Returns the row where the relevant column has the max/min value.
    First, pandas_flavor must be installed via https://pypi.org/project/pandas_flavor/  
    args:
        col:on which column to do the aggregation
        agg_:min or max
    """
    return df[df[col]==df[col].agg(agg_)]    

@register_series_method
def topNValExcluded_(serie, n):
    return serie[~serie.isin(serie.nlargest(n).values)]    

@register_dataframe_method 
def duplicateColumnsCount_(df):
    return df.columns.duplicated().sum()
