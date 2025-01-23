import logging
import os
import numpy as np
import pandas as pd
from concepts import Context
from fcapy.context import FormalContext

import constants
from data.files import extract_text_from_pdf, save_df_to_csv
from utils.logging_utils import get_date, init_debug_config
from utils.os_manipulation import exists_or_create

logger = logging.getLogger(__name__)


class TopicFCA:

    def __init__(self, on_server: bool = True):
        init_debug_config(log_filename='topic_fca_', on_server=on_server)

    def save_ctx_as_txt(self, ctx, save_path: str, filename: str = "context"):
        """
        Save the context to a file.
        :param ctx: Context to save
        :param save_path: Save path including '/' at the end
        :param filename: Filename without type extension
        :return: -
        """
        exists_or_create(path=save_path)
        with open(save_path + filename + ".txt", "w") as text_file:
            text_file.write(ctx)
        logging.info(f"Context saved as txt file: {save_path + filename}.txt")

    def load_ctx_from_txt(self, load_path: str, filename: str = "context"):
        """
        Load the context from a file.
        :param load_path: Load path including '/' at the end
        :param filename: Filename without type extension
        :return: Context
        """
        with open(load_path + filename + ".txt", "r") as text_file:
            read_string_context = text_file.read()
            logging.info(f"loaded context from txt file")
            return Context.fromstring(read_string_context)

    def csv2ctx(self, path_to_file: str, filename: str, prefix: str = "doc_"):
        """
        Load a context from a csv file.
        The entries in the csv file are expected to be 0 or 1.
        They are converted to False or True.
        Moreover, the index column and the first row are converted to the object and attribute names (of type string).
        Since one library is used to read the csv file and another one to work with the context, the context is saved to a
        txt file and loaded again using the second python library.

        :param path_to_file: Path to the csv file including the '/' at the end
        :param filename: Complete filename of the csv file including the type extension
        :param prefix: Prefix of the object ids; might be either "doc_" or "term_"
        :return: Formal context
        """
        df = pd.read_csv(path_to_file + filename, index_col=0)
        df = df.map(lambda x: True if x == 1 else False)
        df.set_index(np.array([prefix + str(doc_id) for doc_id in df.index.tolist()]), inplace=True)
        df.columns = np.array(["topic_" + str(topic_id) for topic_id in df.columns.tolist()])

        new_filename = "read_context_" + filename.split(".")[0]
        ctx_version1 = FormalContext.from_pandas(df)

        self.save_ctx_as_txt(ctx_version1.print_data(ctx_version1.n_objects, ctx_version1.n_attributes),
                             save_path=path_to_file, filename=new_filename)
        ctx = self.load_ctx_from_txt(load_path=path_to_file, filename=new_filename)

        return ctx

    def topic2docs(self, ctx, topic_ids: list[int]):
        """
        Get the extent of a topic.
        The extent of a topic is the set of documents that are associated with the topic.
        :param ctx:
        :param topic_ids:
        :return:
        """
        return ctx.extension(["topic_" + str(t_id) for t_id in topic_ids])

    def doc2topics(self, ctx, doc_ids: list[int], prefix: str = "doc_"):
        """
        Get the intent of a document.
        The intent of a document is the set of topics that are associated with the document.
        :param ctx: Formal context
        :param doc_ids: List of document ids
        :param prefix: Prefix of the document ids; might be either "doc_" or "term_"
        :return:
        """
        return ctx.intension([prefix + str(d_id) for d_id in doc_ids])

    def print_stats(self, ctx):
        """
        Print some statistics of the context.
        :param ctx: Formal context
        :return: -
        """
        print("Number of objects: ", len(ctx.objects))
        print("Number of attributes: ", len(ctx.properties))

    def print_in_extents(self, ctx):
        for extent, intent in ctx.lattice:
            print('%r %r' % (extent, intent))

    def ctx2fimi(self, ctx, path_to_file: str, filename: str = "context_format_fimi", prefix: str = "doc_"):
        """
        Convert a context to a file in the FIMI format.
        According to the FIMI format, each line represents an object.
        The line contains a list of its attributes/features.

         cf. https://fcalgs.sourceforge.net/pcbo-amai.html, https://fcalgs.sourceforge.net/format.html
        :param ctx: Context to convert
        :param path_to_file: Path to save the file including the '/' at the end
        :param filename: Name of the file without type extension
        :param prefix: Prefix of the object ids; might be either "doc_" or "term_"
        :return: -
        """
        # TODO: Works only for doc-topic context, NOT for term-topic context! Add this case.
        exists_or_create(path=path_to_file)
        with open(path_to_file + filename + ".fimi", "x") as f:
            for object_id in range(len(ctx.objects)):
                f.write(f"{' '.join(map(str, list(self.doc2topics(ctx, doc_ids=[object_id], prefix=prefix))))}\n")
        f.close()
        logging.info(f"Context saved as FIMI file: {path_to_file + filename}.fimi")

    def topics2integers(self, path2fimi: str, save_path: str):
        """
        Convert the topics in a FIMI file to integers.
        When converting the term-topic context to a FIMI file, the topics are saved as strings (without any numerical
        identifier).
        Since pcbo stops with a segmentation fault, the topics need to be converted to strings.
        :param path2fimi: Path to the FIMI file including the file ending .fimi
        :param save_path: Path to the save the new FIMI file including the filename and file ending .fimi
        :return: -
        """
        # each line in the FIMI file represents an object
        with open(path2fimi, "r") as f:
            lines = f.readlines()
        f.close()

        topic2int_dict = {}  # mapping from topic to integer

        def str_to_int(element: str):
            """
            Convert the element to an integer.
            :param element: String element
            :return: Integer element
            """
            # strip the newline or extra whitespace
            element = element.strip()

            if element not in topic2int_dict:
                topic2int_dict[element] = len(topic2int_dict)
            return topic2int_dict[element]

        with open(save_path, "w") as f:
            for line in lines:
                # split the line into elements, map them to integers
                converted_line = ' '.join(map(str, map(str_to_int, line.split())))
                f.write(f"{converted_line}\n")
        f.close()
        logging.info(f"Context saved as fimi file: {save_path}")

        # save the mapping to edn file
        exists_or_create(path=''.join(save_path.split('/')[:-1]))
        save_filename = save_path.removesuffix(".fimi") + "_mapping.edn"
        with open(save_filename, "w") as f:
            f.write(str(topic2int_dict))
        logging.info(f"Context converted to integers and saved as edn file: {save_filename}")


    def intents_from_fimi(self, path_to_file: str, filename: str):
        """
        Load intents from a FIMI file.
        :param path_to_file: Path to the FIMI file including the '/' at the end
        :param filename: Complete filename of the FIMI file including the type extension
        :return: List of intents
        """
        with open(path_to_file + filename, "r") as f:
            intents = f.readlines()
        f.close()

        def to_int(element):
            return int(element) if element else None

        return [list(map(to_int, i.removesuffix('\n').split(' '))) for i in intents]

    def reconstruct_concept_from_intent(self, ctx, intent: list[int]):
        """
        Reconstructs a formal concept given its intent.

        :param ctx: List of lists (binary matrix) representing the formal context.
        :param intent: List of attribute indices representing the intent.
        :return: extent, intent_closure representing the reconstructed formal concept.
        """
        extent = self.topic2docs(ctx=ctx, topic_ids=intent if intent[0] else [])
        input_extent = [int(e.split("_")[1]) for e in extent]

        # Compute intent closure (attributes shared by all objects in the extent)
        if extent:  # If the extent is not empty
            intent_closure = set(ctx.properties)  # Start with all attributes
            for obj_idx in input_extent:
                obj_attributes = set(self.doc2topics(ctx=ctx, doc_ids=[obj_idx]))
                intent_closure &= obj_attributes  # Intersect with attributes of the current object
        else:  # If extent is empty, closure is empty
            intent_closure = set()
        return extent, intent_closure

    def get_concept_lattice(self, ctx, intents):
        """
        Get the concept lattice of a context.
        :param ctx: Formal context
        :param intents: List of intents
        :return: Concept lattice as a list of lists, where each inner list represent the extents and intent closures of one
        formal concept.
        """

        return [list(self.reconstruct_concept_from_intent(ctx, input_intent)) for input_intent in intents]

    def obtain_doc_topic_inc_per_subdir(self, parent_path: str, save_path: str, topic_model):
        """
        Obtain the document-topic incidence for each subdirectory in the parent directory.
        If a Subdirectory contains a directory, the function is called recursively.
        :param parent_path: Path to the uppermost directory regarded
        :param save_path: Path to save the document-topic incidence, including the '/' at the end
        :param topic_model: Topic model
        :return:
        """
        logging.info(f"Parent directory: {parent_path}")
        date = get_date()

        # obtain all subdirectories
        for current_directory, subdirectories, files in os.walk(parent_path):
            parent_dir_name = current_directory.split("/")[-1] if "/" in current_directory else current_directory
            logging.info(f"Current directory: {parent_dir_name}; starting now")
            for subdir in subdirectories:
                self.obtain_doc_topic_inc_per_subdir(parent_path=parent_path + subdir + "/", save_path=save_path,
                                                     topic_model=topic_model)  # recursive call

            exists_or_create(save_path + parent_dir_name)
            logging.info(f"Created: {save_path + parent_dir_name}")

            # obtain the document-topic incidence for the current directory
            # obtain texts
            text_files = [current_directory + path for path in files if
                          (path.endswith(('.txt', '.pdf', '.png', '.jpg', '.jpeg')))]
            texts = [extract_text_from_pdf(path, find_caption=True)[0] for path in text_files]
            logging.info(f"Obtained texts for {parent_dir_name}")

            doc_topic_incidence = topic_model.get_document_topic_incidence(doc_ids=np.arange(len(texts)))
            save_df_to_csv(df=doc_topic_incidence, path=save_path + parent_dir_name,
                           file_name=f"{parent_dir_name}_doc_topic_incidence_{date}")
            logging.info(f"Obtained & saved doc-topic incidence for {parent_dir_name}")

            # determine optimal threshold for document-topic incidence
            threshold, row_norm_doc_topic_df = topic_model.determine_threshold_doc_topic_threshold(doc_topic_incidence,
                                                                                                   opt_density=0.1,
                                                                                                   save_path=save_path + parent_dir_name)
            logging.info(f"Optimal threshold for {parent_dir_name}: {threshold}")
            thres_row_norm_doc_topic_df = topic_model.apply_threshold_doc_topic_incidence(row_norm_doc_topic_df,
                                                                                          threshold=threshold)
            save_df_to_csv(df=thres_row_norm_doc_topic_df, path=save_path + parent_dir_name,
                           file_name=f"{parent_dir_name}_thres_row_norm_doc_topic_incidence_{date}")
            logging.info(f"Obtained & saved thresholded doc-topic incidence for {parent_dir_name} under path: {save_path + parent_dir_name}{parent_dir_name}_thres_row_norm_doc_topic_incidence_{date}")

            # translate docs IDs to document names
            translated_thres_row_norm_doc_topic_incidence = thres_row_norm_doc_topic_df.rename(
                index={i: text_files[i] for i in range(len(text_files))})
            save_df_to_csv(df=translated_thres_row_norm_doc_topic_incidence, path=save_path + parent_dir_name,
                           file_name=f"{parent_dir_name}_translated_thres_row_norm_doc_topic_incidence_{date}")
            logging.info(f"Translated doc IDs to document names for {parent_dir_name}.\nSaved under path: {save_path + parent_dir_name}{parent_dir_name}_translated_thres_row_norm_doc_topic_incidence_{date}")

            # term-topic incidence
            save_path_topic_words = save_path + parent_dir_name + f"/{parent_dir_name}_topic2terms_{date}.json"
            term_topic_incidence = topic_model.get_term_topic_incidence(doc_ids=np.arange(len(texts)),
                                                                        save_path_topic_words=save_path_topic_words)
            save_df_to_csv(term_topic_incidence, save_path + parent_dir_name,
                           f"{parent_dir_name}_term_topic_incidence_{date}")
            logging.info(f"Obtained & saved term-topic incidence for {parent_dir_name} under path: {save_path_topic_words}")
            logging.info(f"Finished {parent_dir_name}")

# if __name__ == '__main__':
#     on_server = True
#     date = "19_01_25"
#     path = constants.SERVER_PATH + '/Vehicles/' if on_server else "/Users/klara/Downloads/KDE_Projekt/sample_data_server/"
# #     dataset_path = constants.SERVER_PATH_TO_PROJECT + 'dataset/' if on_server else "../dataset/"
#     model_path = constants.SERVER_PATH_TO_PROJECT + 'models/' if on_server else '../models/'
#     incidence_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/incidences/server_080125/' \
#         if on_server else "../results/incidences/"
#     plot_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/plots/server_080125/' \
#         if on_server else "../results/plots/"
#     top_doc_filename = f"thres_row_norm_doc_topic_incidence{date}.csv" \
#         if on_server else "thres_row_norm_doc_topic_incidence.csv"
#     term_topic_filename = f"term_topic_incidence{date}.csv" \
#         if on_server else "term_topic_incidence.csv"
#
#     # Load the doc-topic context
#     doc_topic_ctx = csv2ctx(path_to_file=incidence_save_path, filename=top_doc_filename)
#     ctx2fimi(doc_topic_ctx, path_to_file=incidence_save_path)
#
#     # Load the term-topic context
#     term_topic_ctx = csv2ctx(path_to_file=incidence_save_path, filename=term_topic_filename)
#     ctx2fimi(term_topic_ctx, path_to_file=incidence_save_path)
#
#     model = tm.TopicModel(documents=None)
#     model.load_model(path=model_path, filename='01_16_25topic_model_01_17_25' if on_server else 'topic_model')
#     obtain_doc_topic_inc_per_subdir(parent_path=path,
#                                     save_path='/norgay/bigstore/kgu/dev/text_topic/results/incidences/190125/' if on_server else '/Users/klara/Downloads/tmp_res',
#                                     topic_model=model, date=date)

# Reconstruct the concept
# top_doc_intents = intents_from_fimi(path_to_file=incidence_save_path, filename=f"doc_topic_intents_{date}.fimi")
# # print("Intents: ", intents)
# print("size of concept lattice: ", len(top_doc_intents))  # == 14476 if local

# visualization
# for input_intent in input_intents[50:]:
#     extent, intent_closure = reconstruct_concept_from_intent(ctx, intents)
#
#     # Output
#     print(f"Given Intent: {input_intent}")
#     print(f"Reconstructed Extent: {extent}")
#     print(f"Reconstructed Intent Closure: {intent_closure}")
# concept_lattice = get_concept_lattice(ctx, intents[50:60])
# print("size of concept lattice: ", len(concept_lattice))  # should be 14476
# print("Concept lattice: ", concept_lattice)

# print_in_extents(ctx=ctx)

# ctx.lattice.graphviz()

# print("Extent of topic 0: ", topic2docs(ctx=ctx, topic_ids=[0]))
#
# print("Intent of doc 0: ", doc2topics(ctx=ctx, doc_ids=list(range(0,30))))
#
# print_stats(ctx)

# test fimi2int
# path2fimi = "/Users/klara/Developer/Uni/WiSe2425/text_topic/results/incidences/context_format_fimi.fimi"
# save_path = "/Users/klara/Developer/Uni/WiSe2425/text_topic/results/incidences/test.fimi"
# topics2integers(path2fimi, save_path)
