import os
from glob import glob

from elasticsearch import Elasticsearch

from data import files
from data.caption_images import ImageCaptioner
from visualization.two_d_display import scatter_documents_2d
from visualization.visualize_stats import stats_as_bar_charts
from visualization.visualize_named_entity_clusters import display_NE_cluster
import constants
import datetime
import tqdm

if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')

    # 2D scatter plot of the documents colored by their parent directory
    # client = Elasticsearch(constants.CLIENT_ADDR)
    # scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, reducer='UMAP', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, reducer='TSNE', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, reducer='PCA', unique_id_suffix=date)

    # visualize data stats
    # base_path2csv = constants.SERVER_STATS_PATH
    # csv_files = files.get_files(path=base_path2csv, file_type='csv')
    # for i in tqdm.tqdm(range(len(csv_files)), desc='Producing bar charts of statistic files'):
    #     csv_file = csv_files[i]
    #     stats_as_bar_charts(path2csv=csv_file, save_path=constants.SERVER_SAVE_PATH + '/16_01_25',
    #                         unique_id_suffix=date + '_' + str(i))

    # visualize named entity clusters
    # save_path = constants.SERVER_SAVE_PATH + "/cluster_NER/"
    # json_files = files.get_files(path=save_path, file_type='json')
    # for reducer in ['TSNE', 'PCA', 'UMAP']:
    #     print(f"Started with reducer: {reducer}")
    #     for i in tqdm.tqdm(range(len(json_files)), desc='Producing plots of NER cluster files'):
    #         file_path = json_files[i]
    #         category = file_path.split('/')[-1].split('_')[3]
    #         ne_cluster_dict = files.load_dict_from_json(path=file_path)
    #         display_NE_cluster(ne_results=ne_cluster_dict, reducer=reducer, category=category,
    #                            save_path=save_path)


    # test captioning images
    local = False
    path2imgs = "/Users/klara/Downloads/" if local else '/norgay/bigstore/kgu/dev/text_topic/results/plots/'
    save_path = "/Users/klara/Downloads/captions/" if local else '/norgay/bigstore/kgu/dev/text_topic/results/captions/'

    # Load the processor and model
    captioner = ImageCaptioner()

    # Load and preprocess the image
    l_img_paths = sorted(glob(os.path.join(path2imgs, "*.png")))
    for image_path in tqdm.tqdm(l_img_paths):
        # print("start image", image_path)
        caption = captioner.caption_image(image_path)
        captioner.save_caption_to_file(image_path, save_path=save_path)
        # print("Generated Caption:", caption)
