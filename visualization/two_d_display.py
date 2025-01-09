import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch
from constants import *
from sklearn.decomposition import PCA
import pandas as pd
from utils.os_manipulation import save_or_not
from sklearn.manifold import TSNE
import numpy as np
import re

def process_directory_names(dir_list:list):
    """
    Process directory names to remove numbers and replace similar names with a common base name.
    :param dir_list: A list of directory names.
    :return: A list of processed directory names, where similar names are replaced with a common base name and
    words with >50% numbers are replaced with the word 'numbers'.
    """
    def is_mostly_numbers(s):
        """Check if a string has more than 50% digits."""
        num_count = sum(c.isdigit() for c in s)
        return num_count > len(s) / 2

    def remove_numbers(s):
        """Remove digits from a string."""
        return re.sub(r'\d+', '', s)

    # Replace strings with >50% numbers with the word 'numbers'
    processed_dirs = []
    for dir_name in dir_list:
        if is_mostly_numbers(dir_name):
            processed_dirs.append("numbers")
        else:
            processed_dirs.append(dir_name)

    # Replace strings differing only by numbers with their shared base word
    shared_dirs = {}
    for dir_name in processed_dirs:
        if dir_name == "numbers":
            shared_dirs[dir_name] = "numbers"
            continue
        base_name = remove_numbers(dir_name)
        shared_dirs[dir_name] = base_name

    # Generate the final list using shared directory names
    final_dirs = [shared_dirs[dir_name] for dir_name in processed_dirs]
    return final_dirs


def scatter_documents_2d(client, save_path=None, on_server=False, use_tsne=False, preprocess_dirs=True,
                         unique_id_suffix=""):
    '''
    This function creates a 2D scatter plot of the documents.
    The documents are represented by their embeddings.
    Each document is colored according to its parent directory.
    The plot is saved as a .png file if save_path is not None.
    :param client: Elasticsearch client
    :param save_path: path to save the plot
    :param on_server: if the function is run on the server. Since there are deeper directories on the server,
    the documents are coloured according to the uppermost directory.
    :param use_tsne: if True, t-SNE is used for dimensionality reduction, else PCA
    :param preprocess_dirs: if True, preprocess the directory names, i.e. remove numbers
    and replace similar names with a common base name
    :param unique_id_suffix: unique id suffix for the file name; default is empty string (hence, no suffix)
    :return -
    '''
    # obtain results from elastic search
    upper_request_limit = 10000
    client.indices.refresh(index=DB_NAME)
    count = int(client.cat.count(index=DB_NAME, format="json")[0]["count"])
    results = []

    res = client.search(index=DB_NAME, body={
        'size': upper_request_limit,  # Number of documents to fetch
        'query': {
            'match_all': {}
        }
    },
                            scroll='2m'  # Keep the scroll context alive for 2 minutes -> used for big data
                            )
    # Get the first batch of results
    scroll_id = res['_scroll_id']
    results.extend(res['hits']['hits'])

    # Fetch additional batches
    while True:
        response = client.scroll(
            scroll_id=scroll_id,
            scroll='2m'
        )
        hits = response['hits']['hits']
        if not hits:  # Break if no more results
            break
        results.extend(hits)  # Add the new results to the list
        scroll_id = response['_scroll_id']  # Update scroll ID for the next iteration

    # Clear the scroll context
    client.clear_scroll(scroll_id=scroll_id)

    embeddings = [r['_source']['embedding'] for r in results]
    class_dirs = [r['_source']['directory'] for r in results]
    # uppermost directory if on server else the directory
    colour_criteria = [r['_source']['path'].split('/ETYNTKE/')[-1].split('/')[0] for r in results] \
        if on_server else class_dirs
    if preprocess_dirs:
        colour_criteria = process_directory_names(colour_criteria)

    # reduce dimensionality to 2D
    transformation = "TSNE" if use_tsne else "PCA"
    if use_tsne:
        tsne = TSNE(n_components=2)
        transformed_embs = tsne.fit_transform(np.array(embeddings))
    else:
        pca = PCA(n_components=2)  # TODO: UMAP
        transformed_embs = pca.fit_transform(embeddings)

    # create dataframe
    df = pd.DataFrame({'x': transformed_embs[:, 0], 'y': transformed_embs[:, 1], 'parent directory': colour_criteria})

    # plot highlighting the different directories as classes
    fig = plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['x'], y=df['y'], hue=df['parent directory'])
    plt.title('2D scatter plot of the documents')
    plt.legend(loc="upper left", fontsize="9", fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(1.04, 1))
    fig.tight_layout()  # if too many classes: will rise a warning, but the plot will be saved correctly
    save_or_not(plt, file_name=f'{transformation}_scatter_documents_dir_2d_{unique_id_suffix}.svg',
                save_path=save_path, format='svg')
    plt.show()

# if __name__ == '__main__':
#     client = Elasticsearch(CLIENT_ADDR)
#     scatter_documents_2d(client, save_path=SERVER_SAVE_PATH)
