import nlp
import spacy

import constants
import topic.topic_modeling as tm
import data.files as files
import database.init_elasticsearch as db
import tqdm
import datetime

from data.files import save_sentences_to_file, save_df_to_csv

if __name__ == '__main__':
    path = constants.SERVER_PATH
    dataset_path = constants.SERVER_PATH_TO_PROJECT + 'dataset/'
    model_path = constants.SERVER_PATH_TO_PROJECT + 'models/'
    incidence_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/incidences/190125/'
    plot_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/plots/'
    date = datetime.datetime.now().strftime('%x').replace('/', '_')
    load_existing_topic_model = False

    print(f"----{date}----on pumbaa (for GPU)\n")
    print("in this case: Don't delete db prior & only insert/update text field to fill db quicker\n")

    # elasticsearch client
    print('Path to the dataset: ', path)
    client = db.insert_caption_texts(client_addr=constants.PUMBAA_CLIENT_ADDR, src_path=constants.SERVER_PATH)
    print("----text (and captions) inserted----")
