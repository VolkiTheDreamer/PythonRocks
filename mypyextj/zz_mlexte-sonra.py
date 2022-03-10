class AutoML:
    def __init__(self, datasettype):
        self.datasettype = datasettype #n for numeric, c for categorical or m for mixed. We are not to infer this due to the fact taht some fatures seem numeric although they should be regarded as categorial????
    
    """
    ***************FIG konusunu hallet, standart olsun*******************
    *************cats,nums dikkat, dfselectdtype olmasın, kullanıcı kolonları kendi versin, gerekise iki kere çalıştsın,
     ilki fikir sahibi olmak için, ikincisi düzeltilmiş haliyle ilişkileri görmek için*******************
     """
    # tarihleri nasıl alacağını belirle
    # nulları nasıl handle edeceğini belirle


    def RunEDA(self,df,cats,nums,i=10,fig=(10,5),otherEdaFunc=False,imputeStrategy=None,outlierThresh=0.25,
    outliertreat="remove", discretizeNums=False):
        """
        args:
            df:dataframe
            cats:categorical features(sometimes, some numeric features could be regarded as categorical.)
            nums:numeric features
            i:features whose cardinality is less than i will be regarded in order to provide a better visualization
            otherEdaFunc:If False, funcs that exist in DataPrep and Pandas-profiling, won't be executed, assuming you already run them
            imputeStrategy:
            outlierThresh:
            outliertreat:
            discretizeNums:for relationfinder(başka ?)
        """
        fm=f"features whose cardinality is less than {i}"
        longseperator="---------------------------------\n\n"

        
        print("Printing Super info")
        print(SuperInfo(df))
        print(longseperator)

        print("<h1>Printing some information for " + fm + "</h1>")
        print("Printing Unique values in " + fm)
        printUniques(df,i)
        print(longseperator)

        print("Printing Value Counts of " + fm)
        printValueCount(df,i)
        print(longseperator)
        
        if otherEdaFunc==True:
            print("plotting countplots for " + fm)
            multicountplot(df,i,fig)
            print(longseperator)


        print("Printing Top N values " + fm)
        ShowTopN(df)
        print(longseperator)

        print("Printing outlier info")
        outlierinfo(df,df.select_dtypes("O").columns,imputeStrategy,thresh)
        print(longseperator)
        #diğer outlier funcs???


        print("Plotting histogram without outliers")
        plotHistWithoutOutliers(df,fig,outlierThresh,imputeStrategy,outliertreat)
        print(longseperator)

        print("Checking if any other null-like value exist")
        findNullLikeValues(df)
        print(longseperator)

        print("plotting nulls")
        nullPlot(df)
        print(longseperator)

        print("plotNumericsBasedOnCategorical")
        plotNumericsBasedOnCategorical(df,cats,nums)
        print(longseperator)



        print("Printing relations between features " + fm)
        if discretizeNums:
            #ayro bi hepler funk yazalım
            pass
        else:
            df2=df
        lr=getListOfRelationsParallel(df2)            
        if (len(lr)>10):
            print(f"too many, ({len(lr)}), relations detected. Run this function(getListOfRelationsParallel) seperately and visualize it")
        else:
            print(lr)


    def Modelling1_Profile(self,df,*,type):
        """
        args:
            type:r for regression, c for classification, s for clustering(segmentation)
        """
        longseperator="---------------------------------\n\n"

        #General

        #For supervised models
        if type=="c": 
            checkIfNumberOfInstanceEnough(df)
            checkIfNumberOFeatures(df)
            checkForImbalancednessForLabels(df)
            remindForSomeProcesses
        elif type=="r": 
            pass
        else:
            pass

    # X and y's are assigned in between, some FE is executed: gerekirse bunu da bi fonk yap
    def Modelling2_AfterXandY(self,X,y):
        pass




        
#importlar
#cython?????
#https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf--->pipeline ile GridSearchCV 
#https://www.scikit-yb.org/en/latest/tutorial.html#the-model-selection-triple
def Profile(df):
    """
        This method gives you whether 
        - outliers exist, if yes the number fo them; 
        - the data skewed
        - data sayısına bak, beli rasyolar v.s, buna göre önerilerde bulunsun: datan az, tdata topla. feature az extaraciton yap. 
        datan yeterli, trans/test yeter, k fold gerekli dğeil.
        en az 10bin if ML.
        sütun sayıs satıra göre fazla ise, over fit edeblir, üstelik eğitim uzun süreiblr
    """
    return 0

def MultiModel(df):
    """

    """
    return 0


def HasFittingProblem(model):
    """
	önce underfitmi diye baksın, accuracy bkelenenden düşükse(bekleneni parmetre ver) devam etmesin
	data sayısı yeterliyese corss-fold!a gerek yok.
    """
    return 0
