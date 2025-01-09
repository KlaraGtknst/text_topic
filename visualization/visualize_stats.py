import datetime
import matplotlib.pyplot as plt
import pandas as pd
import constants


def stats_as_bar_charts(path2csv: str, save_path: str = "", unique_id_suffix: str = ""):
    """
    Generate bar charts for the statistics of the documents in the database.
    :param path2csv: str, path to the csv file containing the statistics
    :param save_path: str, path to save the bar charts
    :param unique_id_suffix: str, unique identifier for the saved bar charts
    """
    df = pd.read_csv(path2csv)
    fig, ax = plt.subplots()
    ax.bar(df['Value'], df['Count'])
    ax.set_title(f"Statistics of the documents_{unique_id_suffix}")
    ax.set_ylabel('Count')
    ax.set_xlabel('Value')
    plt.xticks(fontsize=2)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path != "":
        plt.savefig(save_path + f"bar_chart_{unique_id_suffix}.png")
    plt.show()


if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')

    stats_as_bar_charts(
        path2csv="/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/file_name-stats.csv",
        unique_id_suffix=date, save_path=constants.SAVE_PATH)
