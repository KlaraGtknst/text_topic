import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.plotting_utils import obtain_low_dim_embs


def extract_embs_clusters(ne_results: dict):
    """
    Extract the embeddings clusters from the dictionary.
    :param ne_results: Dictionary containing the embeddings clusters. Must have the keys "top_n_embeddings" and "clusters".
    :return: List of embeddings clusters.
    """
    try:
        return ne_results["top_n_embeddings"], ne_results["clusters"]
    except KeyError as e:
        print("The dictionary must have the keys 'top_n_embeddings' and 'clusters'.")
        raise e


def display_NE_cluster(ne_results: dict, reducer="PCA", category="ORG", save_path=""):
    """
    Display the named entity clusters in a 2D plot.
    :param ne_results: Dictionary containing the named entity clusters.
    :param reducer: if 'TSNE', t-SNE is used for dimensionality reduction, if 'PCA' then PCA is used,
    if 'UMAP' then UMAP is used
    :param category: The category of the named entities.
    :param save_path: Path to save the plot. If "", plot is not saved. Do not include name of the file.
    :return: -
    """
    date = datetime.datetime.now().strftime('%x').replace('/', '_')
    high_dim_embs, clusters = extract_embs_clusters(ne_results)

    # reduce dimensionality to 2D
    transformed_embs = obtain_low_dim_embs(high_dim_embs=high_dim_embs, reducer=reducer)

    # plot the clusters
    plt.figure(figsize=(10, 10))

    plot = sns.scatterplot(x=transformed_embs[:, 0], y=transformed_embs[:, 1], hue=clusters, palette="viridis")
    plot.legend.set_title(f"Named entity clusters (category: {category}, reducer: {reducer})")
    if save_path != "":
        if not save_path.endswith("/"):
            save_path += "/"
        title = f"named_entity_clusters_{category}_{reducer}_{date}.svg"
        plt.savefig(save_path + title, format='svg', dpi=300)
        print(f"Plot saved to {save_path}/{title}.")
    plt.show()



