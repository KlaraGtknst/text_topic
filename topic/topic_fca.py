from concepts import Context
from fcapy.context import FormalContext
import pandas as pd
import numpy as np

def csv2ctx(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.map(lambda x: True if x == 1 else False)
    print(df)
    df.set_index(np.array(["doc_" + str(doc_id) for doc_id in df.index.tolist()]), inplace=True)
    df.columns = np.array(["topic_" + str(topic_id) for topic_id in df.columns.tolist()])

    return FormalContext.from_pandas(df)


if __name__ == '__main__':
    path = "/Users/klara/Documents/uni/"
    dataset_path = "../dataset/"
    model_path = '../models/'
    incidence_save_path = "../results/incidences/"
    plot_save_path = "../results/plots/"
    top_doc_filename = "thres_row_norm_doc_topic_incidence.csv"

    # Load the context
    ctx = csv2ctx(incidence_save_path + top_doc_filename)
    print(ctx)