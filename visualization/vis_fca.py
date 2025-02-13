import logging
import re
import textwrap
import constants
import utils.logging_utils as logging_utils
import utils.os_manipulation as osm
from topic.topic_fca import TopicFCA

logger = logging.getLogger(__name__)


def wrap_label(label, width=10):
    return '\\n'.join(textwrap.wrap(label, width))


def calculate_font_size(label):
    """Dynamically adjusts font size based on label length."""
    default_size = 15
    min_size = 10
    num_words = len(label.split())

    # Reduce font size if label is too long
    if num_words > 20:
        return max(default_size - (num_words // 2), min_size)
    return default_size


def simplify_numbers(label):
    """Finds and replaces increasing sequences of numbers in a label."""
    if 'topic_' in label:
        label = label.replace("topic_", "")

    if any(char not in '0123456789 ' for char in label):    # If label contains non-numeric characters
        return label

    # Extract numbers from label (handles numbers inside words or separated by spaces/commas)
    numbers = re.findall(r'\b\d+\b', label)  # all standalone numbers
    numbers = sorted(set(map(int, numbers)))  # Convert to integers, remove duplicates, and sort

    if not numbers:
        return label  # If no numbers, return original label

    # Detect consecutive sequences and replace with "start-end"
    i = 0
    replacements = []
    while i < len(numbers):
        start = numbers[i]
        while i + 1 < len(numbers) and numbers[i + 1] == numbers[i] + 1:
            i += 1
        end = numbers[i]

        if start != end:
            replacements.append(f"{start}-{end}")
        else:
            replacements.append(str(start))

        i += 1

    # Apply the simplification to the label, process each match in order
    simplified_label = ' '.join(replacements)

    return simplified_label


def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]


def display_context(path2csv: str, save_path: str, filename_of_csv: str, on_server: bool = False,
                    translated: bool = False):
    """
    This function displays the context as a graph and saves it.
    :param path2csv: Path to the csv file that contains the context
    :param save_path: Path to the directory where the graph should be saved. Should end with '/'
    :param filename_of_csv: Filename of the csv file that contains the context
    :param on_server: Boolean indicating whether the code is running on the server or locally
    :param translated: Boolean indicating whether the document/ directory names should translated in the graph
        (i.e. not IDs)
    :return: -
    """

    if ("thres" in filename_of_csv) or ("server-across-dir" in filename_of_csv):
        if ((("translated" in filename_of_csv) and (not translated))  # use translated document/ directory names
                or ((not "translated" in filename_of_csv) and translated)  # use IDs
                or ("term" in filename_of_csv)):
            return

        if not save_path.endswith('/'):
            save_path = save_path + '/'
        topic_fca = TopicFCA(on_server=on_server)

        # Load the context
        # if not on server -> likely to be across-dir-incidence-matrix -> needs space, hence strip prefix
        # else -> likely to be single-dir-incidence-matrix -> no need to strip prefix
        ctx = topic_fca.csv2ctx(path_to_file=path2csv, filename=filename_of_csv, strip_prefix=(not on_server))

        if ctx:
            osm.exists_or_create(path=save_path)
            add_id = filename_of_csv.split("_")[0] if on_server else "across_dirs"
            filename = f"fca_graph_{add_id}_{logging_utils.get_date()}"
            if translated:
                filename += "_translated"

            # Generate the graph object
            dot = ctx.lattice.graphviz(engine='dot', graph_attr={'ranksep': '1.5', 'nodesep': '1.0'})


            # Apply regex-based text wrapping, number simplification, and font size adjustment for node labels
            def replace_label(match):
                label_text = match.group(1)  # Extract label text
                simplified_label = simplify_numbers(label_text)  # Simplify increasing numbers
                wrapped_label = wrap_label(simplified_label, width=30)  # Wrap text; across dir: 40
                font_size = calculate_font_size(simplified_label)  # Adjust font size

                return f'label="{wrapped_label}", fontsize="{font_size}"'  # Apply changes

            # Modify dot.body safely using regex
            dot.body = [re.sub(r'label="(.*?)"', replace_label, line) for line in dot.body]

            # Save the modified Graphviz file
            dot.render(filename=save_path + filename, format='svg', directory=save_path, cleanup=True)


if __name__ == "__main__":
    on_server = False
    date = logging_utils.get_date()
    path2across_dir_csv = "/norgay/bigstore/kgu/dev/clj_exploration_leaks/results/fca-dir-concepts/across-dir/" if (
        on_server) else "/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/fca-dir-concepts/across-dir/"
    save_path = constants.Paths.SERVER_FCA_SAVE_PATH.value + date + '/' if on_server else \
        f"/Users/klara/Developer/Uni/WiSe2425/text_topic/results/fca/{date}/"
    filename_of_csv = "server-across-dir-incidence-matrix.csv"  # "across-dir-incidence-matrix.csv"

    # across-dir-incidence-matrix
    osm.exists_or_create(path=save_path)
    print("Displaying context...")
    display_context(path2csv=path2across_dir_csv, save_path=save_path, filename_of_csv=filename_of_csv,
                    translated=False, on_server=on_server)
