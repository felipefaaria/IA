import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from minisom import MiniSom


def get_data_sample(n_samples=10000):
    #Adicione o arquivo
    df = pd.read_csv('creditcard.csv')
    
    from sklearn.model_selection import train_test_split
    _, df_sample = train_test_split(df, test_size=n_samples, stratify=df['Class'], random_state=42)
    
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    df_sample['scaled_amount'] = scaler.fit_transform(df_sample['Amount'].values.reshape(-1,1))
    df_sample['scaled_time'] = scaler.fit_transform(df_sample['Time'].values.reshape(-1,1))
    df_sample.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    X_sample = df_sample.drop('Class', axis=1).values
    y_sample = df_sample['Class'].values
    
    return X_sample, y_sample

def run_clustering():
    print("--- Preparando Amostra de Dados (10.000 registros) ---")
    X, y = get_data_sample(n_samples=10000)
    print(f"Dados prontos: {X.shape}")

    results = {}

    print("\n--- Executando K-Means ---")

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    sil_kmeans = silhouette_score(X, kmeans_labels)
    ari_kmeans = adjusted_rand_score(y, kmeans_labels)
    
    print(f"K-Means - Silhouette Score: {sil_kmeans:.4f} (Coesão interna)")
    print(f"K-Means - Adjusted Rand Index (ARI): {ari_kmeans:.4f} (Comparação com a Realidade)")
    results['KMeans'] = {'Silhouette': sil_kmeans, 'ARI': ari_kmeans}


    print("\n--- Executando DBSCAN ---")

    dbscan = DBSCAN(eps=3.0, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)
    
    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN encontrou {n_clusters_db} grupos.")
    
    if n_clusters_db > 1:
        sil_db = silhouette_score(X, dbscan_labels)
    else:
        sil_db = -1 
        
    ari_db = adjusted_rand_score(y, dbscan_labels)
    
    print(f"DBSCAN - Silhouette Score: {sil_db:.4f}")
    print(f"DBSCAN - Adjusted Rand Index (ARI): {ari_db:.4f}")
    results['DBSCAN'] = {'Silhouette': sil_db, 'ARI': ari_db}

    print("\n--- Executando SOM (MiniSom) ---")

    som_dim = 10
    som = MiniSom(x=som_dim, y=som_dim, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 1000) 


    q_error = som.quantization_error(X)
    print(f"SOM - Erro de Quantização: {q_error:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()
    plt.title('SOM U-Matrix (Distâncias entre Neurônios)')
    print("Gerando gráfico U-Matrix do SOM...")
    plt.show()
    
    results['SOM'] = {'Quantization Error': q_error}

    return results

if __name__ == "__main__":
    run_clustering()