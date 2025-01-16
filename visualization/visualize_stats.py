import datetime
import matplotlib.pyplot as plt
import pandas as pd
import constants
from utils.os_manipulation import exists_or_create


def stats_as_bar_charts(path2csv: str, save_path: str = "", unique_id_suffix: str = ""):
    """
    Generate bar charts for the statistics of the documents in the database.
    :param path2csv: str, path to the csv file containing the statistics
    :param save_path: str, path to save the bar charts
    :param unique_id_suffix: str, unique identifier for the saved bar charts
    """
    df = pd.read_csv(path2csv)
    fig, ax = plt.subplots()
    try:
        ax.bar(df['Value'], df['Count'])
        type_of_stat = path2csv.split('/')[-1].split('-')[0]
        ax.set_title(f"Statistics of {type_of_stat} (dataset date: {unique_id_suffix})")
        ax.set_ylabel('Count')
        ax.set_xlabel('Value')
        plt.xticks(fontsize=2)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path != "":
            if not save_path.endswith('/'):
                save_path += '/'
            exists_or_create(path=save_path)
            save_path_with_suffix = save_path + f"bar_chart_{type_of_stat}_{unique_id_suffix}.png"
            plt.savefig(save_path_with_suffix, dpi=300)
            print(f"Bar chart saved at: {save_path_with_suffix}")
        plt.show()
    except KeyError as e:
        print(f"Error: {e}. Please check the columns in the csv file: {path2csv}.")


# if __name__ == '__main__':
#     date = datetime.datetime.now().strftime('%x').replace('/', '_')
#
#     stats_as_bar_charts(
#         path2csv="/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/file_name-stats.csv",
#         unique_id_suffix=date, save_path=constants.SAVE_PATH + 'plots/')
