from fcapy.context import FormalContext
from concepts import Context
from fcapy.lattice import ConceptLattice
import pandas as pd
import numpy as np


def save_ctx_as_txt(ctx, save_path:str, filename:str="context"):
    """
    Save the context to a file.
    :param ctx: Context to save
    :param save_path: Save path including '/' at the end
    :param filename: Filename without type extension
    :return: -
    """
    with open(save_path + filename + ".txt", "w") as text_file:
        text_file.write(ctx)

def load_ctx_from_txt(load_path:str, filename:str="context"):
    """
    Load the context from a file.
    :param load_path: Load path including '/' at the end
    :param filename: Filename without type extension
    :return: Context
    """
    with open(load_path + filename + ".txt", "r") as text_file:
        read_string_context = text_file.read()
        return Context.fromstring(read_string_context)

def csv2ctx(path_to_file:str, filename:str):
    """
    Load a context from a csv file.
    The entries in the csv file are expected to be 0 or 1.
    They are converted to False or True.
    Moreover, the index column and the first row are converted to the object and attribute names (of type string).
    Since one library is used to read the csv file and another one to work with the context, the context is saved to a
    txt file and loaded again using the second python library.

    :param path_to_file: Path to the csv file including the '/' at the end
    :param filename: Complete filename of the csv file including the type extension
    :return: Formal context
    """
    df = pd.read_csv(path_to_file + filename, index_col=0)
    df = df.map(lambda x: True if x == 1 else False)
    df.set_index(np.array(["doc_" + str(doc_id) for doc_id in df.index.tolist()]), inplace=True)
    df.columns = np.array(["topic_" + str(topic_id) for topic_id in df.columns.tolist()])

    new_filename = "read_context_" + filename.split(".")[0]
    ctx_version1 = FormalContext.from_pandas(df)

    save_ctx_as_txt(ctx_version1.print_data(ctx_version1.n_objects, ctx_version1.n_attributes),
                    save_path=path_to_file, filename=new_filename)
    ctx = load_ctx_from_txt(load_path=path_to_file, filename=new_filename)

    return ctx

def topic2docs(ctx, topic_ids:list[int]):
    """
    Get the extent of a topic.
    The extent of a topic is the set of documents that are associated with the topic.
    :param ctx:
    :param topic_ids:
    :return:
    """
    return ctx.extension(["topic_" + str(t_id) for t_id in topic_ids])

def doc2topics(ctx, doc_ids:list[int]):
    """
    Get the intent of a document.
    The intent of a document is the set of topics that are associated with the document.
    :param ctx:
    :param doc_ids:
    :return:
    """
    return ctx.intension(["doc_" + str(d_id) for d_id in doc_ids])

def print_stats(ctx):
    """
    Print some statistics of the context.
    :param ctx: Formal context
    :return: -
    """
    # TODO: methods don't exist in the library
    # print("Number of objects: ", ctx.definition())
    # print("Number of attributes: ", ctx.n_attributes)
    # print("Number of relations: ", ctx.n_relations)


def print_in_extents(ctx):
    for extent, intent in ctx.lattice:
        print('%r %r' % (extent, intent))

if __name__ == '__main__':
    path = "/Users/klara/Documents/uni/"
    dataset_path = "../dataset/"
    model_path = '../models/'
    incidence_save_path = "../results/incidences/"
    plot_save_path = "../results/plots/"
    top_doc_filename = "thres_row_norm_doc_topic_incidence.csv"

    # Load the context
    ctx = csv2ctx(path_to_file=incidence_save_path, filename=top_doc_filename)
    #print(ctx)

    #print_in_extents(ctx=ctx)

    #ctx.lattice.graphviz()

    print("Extent of topic 0: ", topic2docs(ctx=ctx, topic_ids=[0]))

    print("Intent of doc 0: ", doc2topics(ctx=ctx, doc_ids=list(range(0,30))))

    print_stats(ctx)



