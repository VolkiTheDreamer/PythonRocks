from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_samples,silhouette_score,calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm,tqdm_notebook 

def run_grid_parallel(tpl):
    cols,param_grid,X=tpl
    results=[]

    for p in tqdm_notebook(ParameterGrid(param_grid[0])):#Kmeans
        X_scl=p["scl"].fit_transform(X)
        wm=np.ones(10)
        wm[[3,9]]=p["wgh"]
        X_scl_weighted=X_scl*wm
        X_scl_weighted=X_scl_weighted[:,cols]
        clu = KMeans(n_clusters=p["nc"], init='k-means++', max_iter=p["max_iter"], n_init=p["n_init"], random_state=0)
        clu.fit(X_scl_weighted)
    #     print(p)
        try:
            results.append((tuple(cols), "KMeans", str(p), silhouette_score(X_scl_weighted, clu.labels_),calinski_harabasz_score(X_scl_weighted, clu.labels_), davies_bouldin_score(X_scl_weighted, clu.labels_),clu.labels_))
        except:
            print(f"Didn't work for params:{p} in Kmeans")


    for p in tqdm_notebook(ParameterGrid(param_grid[1])): #DBSCAN
        X_scl=p["scl"].fit_transform(X)
        wm=np.ones(10)
        wm[[3,9]]=p["wgh"]
        X_scl_weighted=X_scl*wm 
        X_scl_weighted=X_scl_weighted[:,cols]
        clu=DBSCAN(eps=p["eps"],min_samples=p["min_samples"],n_jobs=-1)
        clu.fit(X_scl_weighted)
    #     print(p)
        try:
            results.append((tuple(cols),"DBSCAN", str(p), silhouette_score(X_scl_weighted, clu.labels_),calinski_harabasz_score(X_scl_weighted, clu.labels_), davies_bouldin_score(X_scl_weighted, clu.labels_),clu.labels_))
        except:
            print(f"Didn't work for params:{p} in DBSCAN")


    for p in tqdm_notebook(ParameterGrid(param_grid[2])): #Agglomerative
        X_scl=p["scl"].fit_transform(X)
        wm=np.ones(10)
        wm[[3,9]]=p["wgh"]
        X_scl_weighted=X_scl*wm  
        X_scl_weighted=X_scl_weighted[:,cols]
        clu=AgglomerativeClustering(n_clusters=p["nc"], linkage=p["linkage"])
        clu.fit(X_scl_weighted)
    #     print(p)
        try:
            results.append((tuple(cols),"Agglomerative", str(p),silhouette_score(X_scl_weighted, clu.labels_),calinski_harabasz_score(X_scl_weighted, clu.labels_), davies_bouldin_score(X_scl_weighted, clu.labels_),clu.labels_))
        except:
            print(f"Didn't work for params:{p} in Agglomerative")

    resultDF=pd.DataFrame(results, columns=["Cols", "Algo","params","silhoute","calinski_harabasz","davie_bouldin","labels"])
    return resultDF