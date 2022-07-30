import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,binarize
from sklearn.pipeline import Pipeline 
import os, sys, site
import itertools    
from numpy.random import uniform
from random import sample, seed
from math import isnan
from multiprocessing import Pool
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import learning_curve
import networkx as nx
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
import warnings
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf

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

    
def printScores(y_test,y_pred,x=None,*, alg_type='c',f1avg=None):
    """    
    prints the available performanse scores.
    Args:
    alg_type: c for classfication, r for regressin
    f1avg: if None, taken as binary.
    """
    if alg_type=='c':
        acc=accuracy_score(y_test,y_pred)
        print("Accuracy:",acc)
        recall=recall_score(y_test,y_pred)
        print("Recall:",recall)
        precision=precision_score(y_test,y_pred)
        print("Precision:",precision)
        if f1avg is None:
            f1=f1_score(y_test,y_pred)
        else:
            f1=f1_score(y_test,y_pred,average=f1avg)
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

def drawEpsilonDecider(data,n):
    """
    for DBSCAN
    n: # of neighbours(in the nearest neighbour calculation, the point itself will appear as the first nearest neighbour. so, this should be
    given as min_samples+1.
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

    
       

def get_feature_names_from_columntransformer(ct):
    """
        returns feature names in a dataframe passet to a column transformer. Useful if you have lost the names due to conversion to numpy.
        if it doesn't work, try out the one at https://johaupt.github.io/blog/columnTransformer_feature_names.html or at https://lifesaver.codes/answer/cannot-get-feature-names-after-columntransformer-12525
    """
    final_features=[]
    try:
        
        for trs in ct.transformers_:
            trName=trs[0]
            trClass=trs[1]
            features=trs[2]
            if isinstance(trClass,Pipeline):   
                n,tr=zip(*trClass.steps)
                for t in tr: #t is a transformator object, tr is the list of all transoformators in the pipeline                
                    if isinstance(t,OneHotEncoder):
                        for f in t.get_feature_names_out(features):
                            final_features.append("OHE_"+f) 
                        break
                else: #if not found onehotencoder, add the features directly
                    for f in features:
                        final_features.append(f)                
            elif isinstance(trClass,OneHotEncoder): #?type(trClass)==OneHotEncoder:
                for f in trClass.get_feature_names_out(features):
                    final_features.append("OHE_"+f) 
            else:
                #remainders
                if trName=="remainder":
                    for i in features:
                        final_features.append(ct.feature_names_in_[i])
                #all the others
                else:
                    for f in features:
                        final_features.append(f)                
    except AttributeError:
        print("Your sklearn version may be old and you may need to upgrade it via 'python -m pip install scikit-learn -U'")

    return final_features 

def featureImportanceEncoded(feature_importance_array,feature_names,figsize=(8,6)):
    """
        plots the feature importance plot.
        feature_importance_array:feature_importance_ attribute
    """
    plt.figure(figsize=figsize)
    dfimp=pd.DataFrame(feature_importance_array.reshape(-1,1).T,columns=feature_names).T
    dfimp.index.name="Encoded"
    dfimp.rename(columns={0: "Importance"},inplace=True)
    dfimp.reset_index(inplace=True)
    dfimp["Feature"]=dfimp["Encoded"].apply(lambda x:x[4:].split('_')[0] if "OHE" in x else x)
    dfimp.groupby(by='Feature')["Importance"].sum().sort_values().plot(kind='barh');
    
    
def compareEstimatorsInGridSearch(gs,tableorplot='plot',figsize=(10,5),est="param_clf"):
    """
        Gives a comparison table/plot of the estimators in a gridsearch.
    """
    cvres = gs.cv_results_
    cv_results = pd.DataFrame(cvres)
    cv_results[est]=cv_results[est].apply(lambda x:str(x).split('(')[0])
    cols={"mean_test_score":"MAX of mean_test_score","mean_fit_time":"MIN of mean_fit_time"}
    summary=cv_results.groupby(by=est).agg({"mean_test_score":"max", "mean_fit_time":"min"}).rename(columns=cols)
    summary.sort_values(by='MAX of mean_test_score', ascending=False,inplace=True)
    
    
    if tableorplot=='table':
        return summary
    else:
        _, ax1 = plt.subplots(figsize=figsize)
        color = 'tab:red'
        ax1.xaxis.set_ticks(range(len(summary)))
        ax1.set_xticklabels(summary.index, rotation=45,ha='right')
                
        ax1.set_ylabel('MAX of mean_test_score', color=color)
        ax1.bar(summary.index, summary['MAX of mean_test_score'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0,summary["MAX of mean_test_score"].max()*1.1)

        ax2 = ax1.twinx() 
        color = 'tab:blue'
        ax2.set_ylabel('MIN of mean_fit_time', color=color) 
        ax2.plot(summary.index, summary['MIN of mean_fit_time'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)        
        ax2.set_ylim(0,summary["MIN of mean_fit_time"].max()*1.1)

        plt.show()      

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Depreceated. use 'sklearn.metrics.ConfusionMatrixDisplay(cm).plot();'
    """
    warnings.warn("use 'sklearn.metrics.ConfusionMatrixDisplay(cm).plot();'")      
    
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


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=42
):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    The plot in the second column shows the times required by the models to train 
    with various sizes of training dataset. The plot in the third columns shows 
    how much time was required to train the models for each training sizes.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        random_state=random_state
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.show()


def drawNeuralNetwork(layers,figsize=(10,8)):
    """
        Draws a represantion of the neural network using networkx.
        layers:list of the # of layers including input and output.
    """    
    plt.figure(figsize=figsize)
    pos={}    
    for e,l in enumerate(layers):
        for i in range(l):
            pos[str(l)+"_"+str(i)]=((e+1)*50,i*5+50)


    X=nx.Graph()
    nx.draw_networkx_nodes(X,pos,nodelist=pos.keys(),node_color='r')
    X.add_nodes_from(pos.keys())

    edgelist=[] #list of tuple
    for e,l in enumerate(layers):
        for i in range(l):
            try:
                for k in range(layers[e+1]):
                    try:
                        edgelist.append((str(l)+"_"+str(i),str(layers[e+1])+"_"+str(k)))
                    except:
                        pass
            except:
                    pass


    X.add_edges_from(edgelist)
    for n, p in pos.items():
        X.nodes[n]['pos'] = p    

    nx.draw(X, pos);    

def plotROC(y_test,X_test,estimator,pos_label=1,figsize=(6,6)):
    cm = confusion_matrix(y_test, estimator.predict(X_test))    
    fpr, tpr, _ = roc_curve(y_test, estimator.predict_proba(X_test)[:,1],pos_label=pos_label)
    roc_auc = auc(fpr, tpr) #or roc_auc_score(y_test, y_scores)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label='(ROC-AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'ro', markersize=8, label='Decision Point(Optimal threshold)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(1-sepecifity)')
    plt.ylabel('True Positive Rate(Recall/Sensitivity)')
    plt.title('ROC Curve (TPR vs FPR at each probability threshold)')
    plt.legend(loc="lower right")
    plt.show();

def plot_precision_recall_curve(y_test_encoded,X_test,estimator,threshs=np.linspace(0.0, 0.98, 40),figsize=(16,6)):
    """
        y_test should be labelencoded.
    """
    pred_prob = estimator.predict_proba(X_test)    
    precision, recall, thresholds = precision_recall_curve(y_test_encoded, pred_prob[:,1])
    pr_auc = auc(recall, precision)    

    Xt = [] ; Yp = [] ; Yr = [] 
    for thresh in threshs:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                y_pred = binarize(pred_prob, threshold=thresh)[:,1]
                Xt.append(thresh)
                Yp.append(precision_score(y_test_encoded, y_pred))
                Yr.append(recall_score(y_test_encoded, y_pred))
            except Warning as e:
                print(f"{thresh:.2f}, error , probably division by zero")

        
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.plot(Xt, Yp, "--", label='Precision', color='red')
    plt.plot(Xt, Yr, "--", label='Recall', color='blue')
    plt.title("Precision vs Recall based on decision threshold")
    plt.xlabel('Decision threshold') ; plt.ylabel('Precision - Recall')
    plt.legend()
    plt.subplot(122)
    plt.step(Yr, Yp, color='black', label='LR (PR-AUC = %0.2f)' % pr_auc)
    # calculate the no skill line as the proportion of the positive class (0.145)
    no_skill = len(y_test_encoded[y_test_encoded==1]) / len(y_test_encoded)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='green', label='No Skill')
    # plot the perfect PR curve
    plt.plot([0, 1],[1, 1], color='blue', label='Perfect')
    plt.plot([1, 1],[1, len(y_test_encoded[y_test_encoded==1]) / len(y_test_encoded)], color='blue')
    plt.title('PR Curve')
    plt.xlabel('Recall: TP / (TP+FN)') ; plt.ylabel('Precison: TP / (TP+FP)')
    plt.legend(loc="upper right")
    plt.show();


def find_best_cutoff_for_classification(estimator, y_test_le, X_test, costlist,threshs=np.linspace(0., 0.98, 20)):    
    """
    y_test should be labelencoded as y_test_le
    costlist=cost list for TN, TP, FN, FP
    """
    y_pred_prob = estimator.predict_proba(X_test)
    y_pred = estimator.predict(X_test)
    Xp = [] ; Yp = [] # initialization

    print("Cutoff\t Cost/Instance\t Accuracy\t FN\t FP\t TP\t TN\t Recall\t Precision F1-score")
    for cutoff in threshs:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                y_pred = binarize(y_pred_prob, threshold=cutoff)[:,1]
                cm = confusion_matrix(y_test_le, y_pred)
                TP = cm[1,1]
                TN = cm[0,0]
                FP = cm[0,1]
                FN = cm[1,0]
                cost = costlist[0]*TN + costlist[1]*TP + costlist[2]*FN + costlist[3]*FP
                cost_per_instance = cost/len(y_test_le)
                Xp.append(cutoff)
                Yp.append(cost_per_instance)
                acc=accuracy_score(y_test_le, y_pred)
                rec = cm[1,1]/(cm[1,1]+cm[1,0])
                pre = cm[1,1]/(cm[1,1]+cm[0,1])
                f1  = 2*pre*rec/(pre+rec)
                print(f"{cutoff:.2f}\t {cost_per_instance:.2f}\t\t {acc:.3f}\t\t {FN}\t {FP}\t {TP}\t {TN}\t {rec:.3f}\t {pre:.3f}\t   {f1:.3f}")
            except Warning as e:
                print(f"{cutoff:.2f}\t {cost_per_instance:.2f}\t\t {acc:.3f}\t\t {FN}\t {FP}\t {TP}\t {TN}\t error might have happened from here anywhere")

    plt.figure(figsize=(10,6))
    plt.plot(Xp, Yp)
    plt.xlabel('Threshold value for probability')
    plt.ylabel('Cost per instance')
    plt.axhline(y=min(Yp), xmin=0., xmax=1., linewidth=1, color = 'r')
    plt.show();


def plot_gain_and_lift(estimator,X_test,y_test,pos_label="Yes",figsize=(16,6)):
    """    
        y_test as numpy array
        prints the gain and lift values and plots the charts.    
    """
    prob_df=pd.DataFrame({"Prob":estimator.predict_proba(X_test)[:,1]})
    prob_df["label"]=np.where(y_test==pos_label,1,0)
    prob_df = prob_df.sort_values(by="Prob",ascending=False)
    prob_df['Decile'] = pd.qcut(prob_df['Prob'], 10, labels=list(range(1,11))[::-1])

    #Calculate the actual churn in each decile
    res = pd.crosstab(prob_df['Decile'], prob_df['label'])[1].reset_index().rename(columns = {1: 'Number of Responses'})
    lg = prob_df['Decile'].value_counts(sort = False).reset_index().rename(columns = {'Decile': 'Number of Cases', 'index': 'Decile'})
    lg = pd.merge(lg, res, on = 'Decile').sort_values(by = 'Decile', ascending = False).reset_index(drop = True)
    #Calculate the cumulative
    lg['Cumulative Responses'] = lg['Number of Responses'].cumsum()
    #Calculate the percentage of positive in each decile compared to the total nu
    lg['% of Events'] = np.round(((lg['Number of Responses']/lg['Number of Responses'].sum())*100),2)
    #Calculate the Gain in each decile
    lg['Gain'] = lg['% of Events'].cumsum()
    lg['Decile'] = lg['Decile'].astype('int')
    lg['lift'] = np.round((lg['Gain']/(lg['Decile']*10)),2)
    display(lg)

    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.plot(lg["Decile"],lg["lift"],label="Model")
    plt.plot(lg["Decile"],[1 for i in range(10)],label="Random")
    plt.title("Lift Chart")
    plt.legend()
    plt.xlabel("Decile")
    plt.ylabel("Lift")    
    
    plt.subplot(122)
    plt.plot(lg["Decile"],lg["Gain"],label="Model")
    plt.plot(lg["Decile"],[10*(i+1) for i in range(10)],label="Random")
    plt.title("Gain Chart")
    plt.legend()
    plt.xlabel("Decile")
    plt.ylabel("Gain")
    plt.xlim(0,11)
    plt.ylim(0,110)
    plt.show();    

def plot_gain_and_lift_orj(estimator,X_test,y_test,pos_label="Yes"):
    """    
        y_test as numpy array
        prints the gain and lift values and plots the charts.    
    """
    prob_df=pd.DataFrame({"Prob":estimator.predict_proba(X_test)[:,1]})
    prob_df["label"]=np.where(y_test==pos_label,1,0)
    prob_df = prob_df.sort_values(by="Prob",ascending=False)
    prob_df['Decile'] = pd.qcut(prob_df['Prob'], 10, labels=list(range(1,11))[::-1])

    #Calculate the actual churn in each decile
    res = pd.crosstab(prob_df['Decile'], prob_df['label'])[1].reset_index().rename(columns = {1: 'Number of Responses'})
    lg = prob_df['Decile'].value_counts(sort = False).reset_index().rename(columns = {'Decile': 'Number of Cases', 'index': 'Decile'})
    lg = pd.merge(lg, res, on = 'Decile').sort_values(by = 'Decile', ascending = False).reset_index(drop = True)
    #Calculate the cumulative
    lg['Cumulative Responses'] = lg['Number of Responses'].cumsum()
    #Calculate the percentage of positive in each decile compared to the total nu
    lg['% of Events'] = np.round(((lg['Number of Responses']/lg['Number of Responses'].sum())*100),2)
    #Calculate the Gain in each decile
    lg['Gain'] = lg['% of Events'].cumsum()
    lg['Decile'] = lg['Decile'].astype('int')
    lg['lift'] = np.round((lg['Gain']/(lg['Decile']*10)),2)
    display(lg)
    
    plt.plot(lg["Decile"],lg["lift"],label="Model")
    plt.plot(lg["Decile"],[1 for i in range(10)],label="Random")
    plt.title("Lift Chart")
    plt.legend()
    plt.xlabel("Decile")
    plt.ylabel("Lift")
    plt.show();
    
    plt.plot(lg["Decile"],lg["Gain"],label="Model")
    plt.plot(lg["Decile"],[10*(i+1) for i in range(10)],label="Random")
    plt.title("Gain Chart")
    plt.legend()
    plt.xlabel("Decile")
    plt.ylabel("Gain")
    plt.xlim(0,11)
    plt.ylim(0,110)
    plt.show();    

def linear_model_feature_importance(estimator,preprocessor,feature_selector=None,clfreg_name="clf"):
    """
    plots the feature importance, namely coefficients for linear models.
    args:
        estimator:either pipeline or gridsearch/randomizedsearch object
        preprocessor:variable name of the preprocessor, which is a columtransformer
        feature_selector:if there is a feature selector step, its name.
        clfreg_name:name of the linear model, usually clf for a classifier, reg for a regressor        
    """
                
    if feature_selector is not None:
        if isinstance(estimator,GridSearchCV) or isinstance(estimator,RandomizedSearchCV)\
         or isinstance(estimator,HalvingGridSearchCV) or isinstance(estimator,HalvingRandomSearchCV):
            est=estimator.best_estimator_
        elif isinstance(estimator,Pipeline):
            est=estimator
        else:
            print("Either pipeline or gridsearch/randomsearch should be passes for estimator")
            return
        
        selecteds=est[feature_selector].get_support()
        final_features=[x for e,x in enumerate(get_feature_names_from_columntransformer(preprocessor)) if e in np.argwhere(selecteds==True).ravel()]
    else:
        final_features=get_feature_names_from_columntransformer(preprocessor)

    importance=est[clfreg_name].coef_[0]
    plt.bar(final_features, importance)
    plt.xticks(rotation= 45,horizontalalignment="right");    



def gridsearch_to_df(searcher,topN=5):
    """
    searcher: any of grid/randomized searcher objects or their halving versions
    """
    cvresultdf = pd.DataFrame(searcher.cv_results_)
    cvresultdf = cvresultdf.sort_values("mean_test_score", ascending=False)
    cols=[x for x in searcher.cv_results_.keys() if "param_" in x]+["mean_test_score","std_test_score"]
    return cvresultdf[cols].head(topN)   


def getAnotherEstimatorFromGridSearch(gs_object,estimator):
    cvres = gs_object.cv_results_
    cv_results = pd.DataFrame(cvres)
    cv_results["param_clf"]=cv_results["param_clf"].apply(lambda x:str(x).split('(')[0])

    dtc=cv_results[cv_results["param_clf"]==estimator]
    return dtc.getRowOnAggregation_("mean_test_score","max")["params"].values 

def cooksdistance(X,y,figsize=(8,6),ylim=0.5):    
    model = sm.OLS(y,X)
    fitted = model.fit()
    # Cook's distance
    pr=X.shape[1]
    CD = 4.0/(X.shape[0]-pr-1)
    influence = fitted.get_influence()
    #c is the distance and p is p-value
    (c, p) = influence.cooks_distance
    plt.figure(figsize=figsize)
    plt.stem(np.arange(len(c)), c, markerfmt=",")
    plt.axhline(y=CD, color='r')
    plt.ylabel('Cook\'s D')
    plt.xlabel('Observation Number')
    plt.ylim(0,ylim)
    plt.show();
