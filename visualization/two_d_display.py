import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import constants
from utils.os_manipulation import save_or_not
from visualization.plotting_utils import obtain_low_dim_embs


def process_directory_names(dir_list: list, max_num_alphabetic: int = 4, num_threshold: float = 0.5):
    """
    Process directory names to remove numbers and replace similar names with a common base name.
    :param dir_list: A list of directory names.
    :param max_num_alphabetic: The maximum number of alphabetic characters in a string to not be replaced with 'chars'.
    :param num_threshold: The threshold for the percentage of digits in a string to be considered mostly numbers.
    All strings with >50% digits are replaced with 'numbers'.
    :return: A list of processed directory names, where similar names are replaced with a common base name and
    words with >50% numbers are replaced with the word 'numbers'.
    """

    def is_mostly_numbers(s):
        """Check if a string has more than 50% digits."""
        num_count = sum(c.isdigit() for c in s)
        return num_count > len(s) * num_threshold

    def remove_numbers(s):
        """Remove digits from a string."""
        return re.sub(r'\d+', '', s)

    def is_all_alphabet(s):
        """Check if all characters in a string are alphabetic."""
        return all(c.isalpha() for c in s)

    # Replace strings with >50% numbers with the word 'numbers'
    processed_dirs = []
    for dir_name in dir_list:
        if is_mostly_numbers(dir_name):
            # Replace strings with >50% numbers with 'numbers'
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
        shared_dirs[dir_name] = base_name   # may produce short alphabetical strings

    # Replace short alphabetic strings (<4 characters) with 'chars'
    for i in range(len(processed_dirs)):
        dir_name = processed_dirs[i]
        if len(dir_name) < max_num_alphabetic and is_all_alphabet(dir_name):
            processed_dirs[i] = 'chars'

    # Generate the final list using shared directory names
    final_dirs = [shared_dirs[dir_name] for dir_name in processed_dirs]
    return final_dirs


def scatter_documents_2d(client, save_path=None, on_server=False, reducer='PCA', preprocess_dirs=True,
                         unique_id_suffix=""):
    """
    This function creates a 2D scatter plot of the documents.
    The documents are represented by their embeddings.
    Each document is colored according to its parent directory.
    The plot is saved as a .png file if save_path is not None.
    :param client: Elasticsearch client
    :param save_path: path to save the plot
    :param on_server: if the function is run on the server. Since there are deeper directories on the server,
    the documents are coloured according to the uppermost directory.
    :param reducer: if 'TSNE', t-SNE is used for dimensionality reduction, if 'PCA' then PCA is used,
    if 'UMAP' then UMAP is used
    :param preprocess_dirs: if True, preprocess the directory names, i.e. remove numbers
    and replace similar names with a common base name
    :param unique_id_suffix: unique id suffix for the file name; default is empty string (hence, no suffix)
    :return -
    """
    # obtain results from elastic search
    upper_request_limit = 10000
    client.indices.refresh(index=constants.DatabaseAddr.DB_NAME.value)
    results = []

    res = client.search(index=constants.DatabaseAddr.DB_NAME.value, body={
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
    transformed_embs = obtain_low_dim_embs(high_dim_embs=embeddings, reducer=reducer)

    # create dataframe
    df = pd.DataFrame({'x': transformed_embs[:, 0], 'y': transformed_embs[:, 1], 'parent directory': colour_criteria})

    # plot highlighting the different directories as classes
    fig = plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['x'], y=df['y'], hue=df['parent directory'])
    plt.title('2D scatter plot of the documents')
    plt.legend(loc="upper left", fontsize="9", fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(1.04, 1))
    fig.tight_layout()  # if too many classes: will rise a warning, but the plot will be saved correctly
    save_or_not(plt, file_name=f'{reducer}_scatter_documents_dir_2d_{unique_id_suffix}.svg',
                save_path=save_path, format='svg')
    plt.show()
