import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def obtain_low_dim_embs(high_dim_embs: list, reducer: str = 'PCA'):
    """
    Obtain low-dimensional embeddings using PCA, t-SNE or UMAP.
    :param high_dim_embs: High-dimensional embeddings. It should be a list of lists,
    while the inner lists have the same number of elements.
    :param reducer: A string specifying the dimensionality reduction technique. One of 'PCA', 'TSNE', 'UMAP'.
    :return: Transformed embeddings in 2D using the specified dimensionality reduction technique.
    """
    if reducer == 'TSNE':
        tsne = TSNE(n_components=2, n_jobs=1)
        transformed_embs = tsne.fit_transform(np.array(high_dim_embs))
    elif reducer == 'UMAP':
        umap_reducer = umap.UMAP(n_components=2)
        transformed_embs = umap_reducer.fit_transform(high_dim_embs)
    else:  # PCA is the default
        pca = PCA(n_components=2)
        transformed_embs = pca.fit_transform(high_dim_embs)
    return transformed_embs
