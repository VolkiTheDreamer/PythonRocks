import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,roc_auc_score,roc_curve
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline 
import os, sys, site
import itertools    
from numpy.random import uniform
from random import sample
from math import isnan
from multiprocessing import Pool
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity


def printAlgorithm(algo):
    """
    You need the change the path.
    """
    p=os.getcwd()
    os.chdir(r"E:\OneDrive\Dökümanlar\GitHub\PythonRocks")
    df=pd.read_excel("Algorithms.xlsx",skiprows=1)
    print(df[df.Algorithm==algo].T)
    os.chdir(p)

def adjustedr2(R_sq,y,y_pred,x):
    return 1 - (1-R_sq)*(len(y)-1)/(len(y_pred)-x.shape[1]-1)

def calculate_aic_bic(n, mse, num_params):
    """
    n=number of instances in y
    """    
    aic = n *np.log(mse) + 2 * num_params
    bic = n * np.log(mse) + num_params * np.log(n)
    # ssr = fitted.ssr #residual sum of squares
    # AIC = N + N*np.log(2.0*np.pi*ssr/N)+2.0*(p+1)
    # print(AIC)
    # BIC = N + N*np.log(2.0*np.pi*ssr/N) + p*np.log(N)
    # print(BIC)
    return aic, bic   

    



def printScores(y_test,y_pred,x=None,*, alg_type='c'):
    """    
    Args:
    alg_type: c for classfication, r for regressin
    """
    if alg_type=='c':
        acc=accuracy_score(y_test,y_pred)
        print("Accuracy:",acc)
        recall=recall_score(y_test,y_pred)
        print("Recall:",recall)
        precision=precision_score(y_test,y_pred)
        print("Precision:",precision)
        f1=f1_score(y_test,y_pred)
        print("F1:",f1)
        return acc,recall,precision,f1
    else:
        mse=mean_squared_error(y_test,y_pred) #RMSE için squared=False yapılabilir ama bize mse de lazım
        rmse=round(np.sqrt(mse),2)
        print("RMSE:",rmse)
        mae=round(mean_absolute_error(y_test,y_pred),2)
        print("MAE:",mae)        
        r2=round(r2_score(y_test,y_pred),2)
        print("r2:",r2)
        adjr2=round(adjustedr2(r2_score(y_test,y_pred),y_test,y_pred,x),2)
        print("Adjusted R2:",adjr2)
        aic, bic=calculate_aic_bic(len(y_test),mse,len(x))
        print("AIC:",round(aic,2))
        print("BIC:",round(bic,2))
        return (rmse,mae,r2,adjr2,round(aic,2),round(bic,2))

def draw_siluet(range_n_clusters,data,isbasic=True,printScores=True):
    """
    Used for K-means
    """
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
    plt.plot(distances)
    
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
    """
    found here: https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
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
    
def CheckForClusterinTendencyWithHopkins(df):
    """
    taken from https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/
    the closer to 1, the higher probability of clustering tendency
    """
    d = df.shape[1]
    #d = len(vars) # columns
    n = len(df) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(df.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(df,axis=0),np.amax(df,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(df.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H    

def getNumberofCatsAndNumsFromDatasets(path,size=10_000_000):
    """
    returns the number of features by their main type(i.e categorical or numeric or datetime)
    args:
        path:path of the files residing in.
        size:size of the file(default is ~10MB). if chosen larger, it will take longer to return.
    """
    os.chdir(path)
    files=os.listdir()
    liste=[]
    for d in files:  
        try:
            if os.path.isfile(d) and os.path.getsize(d)<size:        
                if os.path.splitext(d)[1]==".csv":
                    df=pd.read_csv(d,encoding = "ISO-8859-1")
                elif os.path.splitext(d)[1]==".xlsx":
                    df=pd.read_excel(d)
                else:            
                    continue      

                nums=len(df.select_dtypes("number").columns)        
                date=len(df.select_dtypes(include=[np.datetime64]).columns)
                cats=len(df.select_dtypes("O").columns)-date
                liste.append((d,nums,cats,date))
        except:
            pass

    dffinal=pd.DataFrame(liste,columns=["filename","numeric","categorical","datettime"])
    dffinal.set_index("filename")
    return dffinal


    

#Functions to run before and during modelling
def checkIfNumberOfInstanceEnough(df):
    """
    o Çok az satır varsa daha fazla veri toplanması sağlanmalıdır
    o Aşırı çok satır varsa kısmi sampling yapılabilir.(Detayları göreceğiz)
    o Data çokluğundan emin değilseniz tamamıyla deneyin. Eğitim süresi çok uzun sürüyorsa aşamalı olarak azaltabilirsiniz.
    """

def checkIfNumberOFeatures(df):
    """
    o Az kolon(feature) varsa yenileri temin edilmeye çalışılabilir
    o Çok kolon varsa çeşitli boyut indirgeme ve önemli kolonları seçme yöntemleri uygulanır(Detayları sorna göreceğiz)
    o Yine satırlardaki aynı mantıkla çok kolon olup olmadığında emin değilseniz önce tümüyle birlikte modelleme yapılır. Eğitim süresi uzun ise veya overfitting oluyorsa feature azaltma yöntemleri uygulanabilir.
    Kolon sayısını azaltma sadece eğitim zamanını kısatlmakla kalmaz aynı zamanda overfittingi de engeller.
    """

def checkForImbalancednessForLabels(df):    
    """
    (Imbalanced ise train/test ayrımından sonra oversample yapılır)
    """

def remindForSomeProcesses():
    """
    ....
    """
    print("transformasyon gerektirmeyen kısımlar: feature extraction, feaute selection, feature elimination")        
    
def remindForDiscreteization():
    """
    yüksek carianlitiy olan numeriklerde hangi durumlarda discretization?
    """
    
#arada X ve y manuel belirlenir    
def traintest(X,y,testsize):
    # önce trasin test yaptır, gerekirse başka parameterler de al
    print("dont touch test set")
    
def remindForStep2FE():
    print("transformasyon gerektiren işlemler step 2, hangileri?????????")

#bu arada aşağıdaki açıklamadaki ilk satır çalışablir
def buildModel(train,test):
    """
        çoklu model mi kursak burda? VotingClassifier. parametre olarak pipelineları mı versek. evetse bi önjceki stepte bunu da hatıratsın, tellWhatAlgorithmsToUse bu da çalışsın tabi
        fit trasnform
        pedicr
        skor kontrolü, çok düşükse underfitting sebeplerine bak, belli bi sebep yoksa yeni feature + yeni veri(azsa), veya yeni model
        skor iyiyse cv kontrol
        test setini ver

    """


def tellWhatAlgorithmsToUse(df,type):
    """
    s ve u için ayrı ayrı mı?
    """    