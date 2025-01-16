import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import constants
from utils.os_manipulation import exists_or_create


def insert_linebreak(label: str, limit_len: int = 48):
    """
    Insert line breaks in the label string.
    https://stackoverflow.com/questions/59466109/how-to-get-x-axis-labels-in-multiple-line-in-matplotlib (16.01.2025)
    :param label: The label string
    :param limit_len: The maximum length of the label before inserting a line break
    :return: A string with line breaks
    """
    return '\n'.join(label[i:i + limit_len] for i in range(0, len(label), limit_len)) if len(label) > limit_len else label


def stats_as_bar_charts(path2csv: str, save_path: str = "", unique_id_suffix: str = ""):
    """
    Generate bar charts for the statistics of the documents in the database.
    :param path2csv: str, path to the csv file containing the statistics
    :param save_path: str, path to save the bar charts
    :param unique_id_suffix: str, unique identifier for the saved bar charts
    """
    df = pd.read_csv(path2csv)
    fig, ax = plt.subplots(figsize=(max(len(df['Value'])//10, 10), 6))
    try:
        ax.bar(df['Value'], df['Count'])
        type_of_stat = path2csv.split('/')[-1].split('-')[0]
        ax.set_title(f"Statistics of {type_of_stat} (date: {unique_id_suffix})")
        ax.set_ylabel('Count')
        ax.set_xlabel('Value')
        # if many values, reduce font size
        fontsize = 6 if len(df['Value']) < 20 else 3.5
        labels = [insert_linebreak(label) for label in df['Value']] if len(df['Value']) < 20 else df['Value']
        plt.xticks(rotation=45, ha='right', fontsize=fontsize, labels=labels, ticks=np.arange(len(df['Value'])))
        plt.tight_layout()

        if save_path != "":
            if not save_path.endswith('/'):
                save_path += '/'
            exists_or_create(path=save_path)
            save_path_with_suffix = save_path + f"bar_chart_{type_of_stat}_{unique_id_suffix}.svg"
            plt.savefig(save_path_with_suffix, dpi=300, bbox_inches='tight', format='svg')
            print(f"Bar chart saved at: {save_path_with_suffix}")
        plt.show()
    except KeyError as e:
        print(f"Error: {e}. Please check the columns in the csv file: {path2csv}.")

if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')

    stats_as_bar_charts(
        path2csv="/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/file_name-stats.csv",
        unique_id_suffix=date, save_path=constants.SAVE_PATH + 'plots/')
