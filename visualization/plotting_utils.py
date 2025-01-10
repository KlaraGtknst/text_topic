import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def obtain_low_dim_embs(high_dim_embs, reducer):
    if reducer == 'TSNE':
        tsne = TSNE(n_components=2)
        transformed_embs = tsne.fit_transform(np.array(high_dim_embs))
    elif reducer == 'UMAP':
        umap_reducer = umap.UMAP(n_components=2)
        transformed_embs = umap_reducer.fit_transform(high_dim_embs)
    else:  # PCA is the default
        pca = PCA(n_components=2)
        transformed_embs = pca.fit_transform(high_dim_embs)
    return transformed_embs