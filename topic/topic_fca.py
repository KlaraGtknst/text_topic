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
    :param ctx: Formal context
    :param doc_ids: List of document ids
    :return:
    """
    return ctx.intension(["doc_" + str(d_id) for d_id in doc_ids])

def print_stats(ctx):
    """
    Print some statistics of the context.
    :param ctx: Formal context
    :return: -
    """
    print("Number of objects: ", len(ctx.objects))
    print("Number of attributes: ", len(ctx.properties))


def print_in_extents(ctx):
    for extent, intent in ctx.lattice:
        print('%r %r' % (extent, intent))

def ctx2fimi(ctx, path_to_file:str):
    """
    Convert a context to a file in the FIMI format.
    According to the FIMI format, each line represents an object.
    The line contains a list of its attributes/features.

     cf. https://fcalgs.sourceforge.net/pcbo-amai.html, https://fcalgs.sourceforge.net/format.html
    :param ctx: Context to convert
    :param path_to_file: Path to save the file including the '/' at the end
    :return: -
    """
    with open(path_to_file + "context_format_fimi.fimi", "x") as f:
        for object_id in range(len(ctx.objects)):
            f.write(f"{' '.join(map(str, list(doc2topics(ctx, doc_ids=[object_id]))))}\n")
    f.close()
    print("Context saved as FIMI file")

def intents_from_fimi(path_to_file:str, filename:str):
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


def reconstruct_concept_from_intent(ctx, intent:list[int]):
    """
    Reconstructs a formal concept given its intent.

    :param ctx: List of lists (binary matrix) representing the formal context.
    :param intent: List of attribute indices representing the intent.
    :return: extent, intent_closure representing the reconstructed formal concept.
    """
    extent = topic2docs(ctx=ctx, topic_ids=intent if intent[0] else [])
    #print("Extent: ", extent)
    input_extent = [int(e.split("_")[1]) for e in extent]
    #intent_closure = doc2topics(ctx=ctx, doc_ids=input_extent)  # reconstruction != original intent???
    # Compute intent closure (attributes shared by all objects in the extent)
    if extent:  # If the extent is not empty
        intent_closure = set(ctx.properties)  # Start with all attributes
        for obj_idx in input_extent:
            obj_attributes = set(doc2topics(ctx=ctx, doc_ids=[obj_idx]))
            intent_closure &= obj_attributes  # Intersect with attributes of the current object
    else:  # If extent is empty, closure is empty
        intent_closure = set()
    #print("Intent Closure: ", intent_closure)
    return extent, intent_closure

def get_concept_lattice(ctx, intents):
    """
    Get the concept lattice of a context.
    :param ctx: Formal context
    :param intents: List of intents
    :return: Concept lattice as a list of lists, where each inner list represent the extents and intent closures of one
    formal concept.
    """

    return [list(reconstruct_concept_from_intent(ctx, input_intent)) for input_intent in intents]




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
    #ctx2fimi(ctx, path_to_file=incidence_save_path)
    #print("Context converted to FIMI format")

    # Reconstruct the concept
    intents = intents_from_fimi(path_to_file=incidence_save_path, filename="intents.fimi")
    #print("Intents: ", intents)
    print("size of concept lattice: ", len(intents))    # == 14476

    # for input_intent in input_intents[50:]:
    #     extent, intent_closure = reconstruct_concept_from_intent(ctx, intents)
    #
    #     # Output
    #     print(f"Given Intent: {input_intent}")
    #     print(f"Reconstructed Extent: {extent}")
    #     print(f"Reconstructed Intent Closure: {intent_closure}")
    concept_lattice = get_concept_lattice(ctx, intents[50:60])
    print("size of concept lattice: ", len(concept_lattice))  # should be 14476
    print("Concept lattice: ", concept_lattice)

    #print_in_extents(ctx=ctx)

    #ctx.lattice.graphviz()

    # print("Extent of topic 0: ", topic2docs(ctx=ctx, topic_ids=[0]))
    #
    # print("Intent of doc 0: ", doc2topics(ctx=ctx, doc_ids=list(range(0,30))))
    #
    # print_stats(ctx)



