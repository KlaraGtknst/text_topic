import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.plotting_utils import obtain_low_dim_embs


def extract_embs_clusters(ne_results: dict):
    """
    Extract the embeddings clusters from the dictionary.
    :param ne_results: Dictionary containing the embeddings clusters. Must have the keys "top_n_embeddings" and "clusters".
    :return: Two lists containing keys (named entities), embeddings, clusters.
    """
    try:
        return (list(ne_results["top_n_embeddings"].keys()), list(ne_results["top_n_embeddings"].values()),
                list(ne_results["clusters"].values()))
    except KeyError as e:
        print("The dictionary must have the keys 'top_n_embeddings' and 'clusters'.")
        raise e

def format_labels(labels: list, clusters: list):
    # Create a dictionary to hold the labels grouped by their cluster
    grouped_labels = {}

    # Group labels by their cluster
    for label, cluster in zip(labels, clusters):
        if cluster not in grouped_labels:
            grouped_labels[cluster] = []
        grouped_labels[cluster].append(label)

    # Convert each group of labels to a string (e.g., comma-separated)
    grouped_labels_strings = {
        cluster: ", ".join(group) for cluster, group in grouped_labels.items()
    }

    return grouped_labels_strings


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
    labels, high_dim_embs, clusters = extract_embs_clusters(ne_results)

    # reduce dimensionality to 2D
    transformed_embs = obtain_low_dim_embs(high_dim_embs=high_dim_embs, reducer=reducer)

    # plot the clusters
    plt.figure(figsize=(10, 10))

    sns.scatterplot(x=transformed_embs[:, 0], y=transformed_embs[:, 1], hue=clusters, palette="viridis")

    # legend for the clusters (with named entity labels)
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=list(format_labels(labels=labels, clusters=clusters).values()),
               title='Named Entities')

    plt.title(f"Named entity clusters (category: {category}, reducer: {reducer})")
    if save_path != "":
        if not save_path.endswith("/"):
            save_path += "/"
        title = f"named_entity_clusters_{category}_{reducer}_{date}.svg"
        plt.savefig(save_path + title, format='svg', dpi=300)
        print(f"Plot saved to {save_path}{title}.")
    plt.show()



