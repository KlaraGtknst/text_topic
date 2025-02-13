import pandas as pd
import pypdf as pdf
import hashlib
import warnings
import tqdm
import logging
import csv
import glob
import json
import os
import utils.os_manipulation as osm
from pdf2image import convert_from_path
from data.caption_images import ImageCaptioner

# Suppress logging from pypdf
logging.getLogger("pypdf").setLevel(logging.CRITICAL)


def load_dict_from_json(path: str):
    """
    This function loads a dictionary from a json file.
    :param path: Path to the json file
    :return: Dictionary
    """
    try:
        with open(path, 'r') as f:
            dictionary = json.load(f)
        return dictionary
    except Exception as e:
        raise e


def get_files(path: str = "/", file_type: str = 'pdf', recursive: bool = True):
    """
    This function returns a list of all file paths that end with 'pdf' in a directory.
    :param path: Path to the directory; if no path is given, the function returns all pdf files in the current directory.
    :param file_type: Type of files to return; default is 'pdf'
    :param recursive: If True, the function returns all files in the directory and its subdirectories
    :return: List of file paths
    """
    if not path.endswith("/"):
        path += "/"
    return [path for path in glob.glob(f"{path}/**", recursive=recursive) if path.endswith(file_type)]


def extract_text_from_txt(path: str):
    """
    This function extracts the text from a txt file.
    :param path: Path to the txt file.
    :return: Text from the txt file (string)
    """
    try:
        with open(path, 'r') as f:
            text = f.read()
        return text, True
    except Exception as e:  # all other errors
        return str(e), False


def extract_text_from_pdf(path: str, find_caption: bool = False):
    """
    This function extracts the text from a pdf file.
    If the pdf file is not readable, the function returns a list which contains the error message.
    :param path: Path to the pdf file
    :param find_caption: If True, the function uses the ImageCaptioner to caption the image of the pdf page,
        if no text can be extracted
    :return: List of text from the pdf file; each entry is the text of one page
    """
    try:
        reader = pdf.PdfReader(path)
        image_captioner = ImageCaptioner()

        text = []
        for i, page in enumerate(reader.pages):

            try:
                # Attempt to extract text
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                elif find_caption:
                    # caption image of page
                    dummy_save_path, image = pdf2png(pdf_path=path, png_path='', page_num=i)
                    caption = image_captioner.caption_image(image)
                    text.append(caption)
                else:
                    text.append(f"Image/ Empty page.")
            except Exception as e:
                text.append(f"Error extracting text from page: {e}")
        return " ".join(text), True
    except pdf.errors.PdfStreamError as e:
        return str(e), False
    except AttributeError as e:  # Document is encrypted
        return str(e), False
    except ValueError as e:  # negative seek value -1
        return str(e), False
    except Exception as e:  # all other errors
        return str(e), False


def pdf_to_str(path: str) -> str:
    """
    :param path: path to pdf file
    :return: text from pdf file

    This function extracts the text from a pdf file.
    cf. https://pypi.org/project/PyPDF2/
    """
    with warnings.catch_warnings(action="ignore"):
        warnings.simplefilter("ignore")  # Ignore all warnings
        try:
            reader = pdf.PdfReader(path)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text

        except Exception as e:
            return ''


def get_hash_file(path: str):
    """
    :param path: path to the file
    :return: hash of the file
    """
    block_size = 65536000  # The size of each read from the file
    file_hash = hashlib.sha256()  # Create the hash object, can use something other than `.sha256()` if you wish
    with open(path, 'rb') as f:  # Open the file to read its bytes, automatically closes file at end
        fb = f.read(block_size)  # Read from the file. Take in the amount declared above
        while len(fb) > 0:  # While there is still data being read from the file
            file_hash.update(fb)  # Update the hash
            fb = f.read(block_size)
    id = file_hash.hexdigest()
    return id


def save_sentences_to_file(sentences, dataset_path, save_filename: str = 'sentences2.txt'):
    """
    This function saves the sentences to a file.
    Each new sentence is preceded by the string 'NEWFILE'.
    :param sentences: List of sentences to save
    :param dataset_path: Path to the dataset; ends with '/'
    :param save_filename: Name of the file to save the sentences to
    :return: -
    """
    osm.exists_or_create(dataset_path)  # create the directory if it does not exist
    with open(dataset_path + save_filename, 'w') as f:
        for i in tqdm.tqdm(range(len(sentences)), desc='Writing sentences to file'):
            sentence = sentences[i].encode("utf-8", errors="ignore")
            try:
                f.write(f"NEWFILE{sentence}")
            except AttributeError as e:
                print(f"Error with sentence {i} encountered: {e}")
                pass
    f.close()


def load_sentences_from_file(dataset_path, filename: str = 'sentences1.txt'):
    """
    This function loads the sentences from a file.
    File was created with save_sentences_to_file.
    :param dataset_path: Path to the dataset; ends with '/'
    :param filename: Name of the file to load the sentences from
    :return: String of sentences
    """
    # load the sentences from a file
    with open(dataset_path + filename) as f:
        sentences = f.read()
    print("File content read successfully")
    return sentences


def save_df_to_csv(df, path, file_name):
    """
    This function saves a dataframe to a csv file.
    :param df: Dataframe to save
    :param path: Path to save the file, incl. / at the end
    :param file_name: Name of the file, without file ending
    :return: -
    """
    osm.exists_or_create(path=path)
    df.to_csv(path + file_name + '.csv', index=True)
    print(f"Dataframe saved to {path}")


def pdf2png(pdf_path: str, png_path: str, page_num: int):
    """
    This function converts a pdf file to a png file.
    :param pdf_path: Path to the pdf file
    :param png_path: Path to save the png file, incl. / at the end
    :param page_num: Number of page to convert (starting with 0)
    :return: save_path: Path to the saved png file
    """
    pillow_pixel_limit = 89478485  # if the image has more pixels, warning of bomb DOS attack
    default_dpi = 300
    try:
        document_name = pdf_path.split('/')[-1].split('.')[0]

        images = convert_from_path(pdf_path, dpi=default_dpi)

        # reduce dpi if number of pixels exceeds limit
        width, height = images[0].size
        total_pixels = width * height
        if total_pixels > pillow_pixel_limit:
            # Calculate new DPI based on pixel limit
            scale_factor = (pillow_pixel_limit / total_pixels) ** 0.5
            lower_dpi = int(default_dpi * scale_factor)
            images = convert_from_path(pdf_path, dpi=lower_dpi)

        save_path = png_path + f'image_{document_name}_{page_num}.jpg'
        if png_path != '':
            osm.exists_or_create(png_path)
            images[page_num].save(save_path, 'JPEG')

        return save_path, images[page_num]

    except Exception as e:
        print(f"Error converting pdf to png: {e}")
        pass


def dir_topic_words2csv(dir_path: str, output_file: str, top_n: int = 5):
    """
    This function converts a json file containing the topics and their top 50 word of  a directory to a CSV file.
    :param dir_path: Path to the directory with the topic words files
    :param output_file: Path to save the CSV file, incl. filename and file ending
    :param top_n: Number of top words to extract from each topic
    :return: -
    """
    with open(dir_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract non-empty topics
    filtered_topics = [(topic_id, ", ".join(words[:top_n])) for topic_id, words in data.items() if words]

    # Write to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Topic ID", "Topic Words"])  # Write header
        writer.writerows(filtered_topics)

    print(f"CSV file saved to {output_file}")


def dir_topic_words2csv_across_dirs(path2single_dirs_json: str, path2across_dir_csv: str, save_path: str,
                                    top_n: int = 5):
    """
    Since the across-directory incidence maps directory names to topics (1 if present, 0 if not), but there is no way
    to obtain the top words for each topic, this function extracts the top words for each topic from the
    single-directory json files and saves them to a CSV file.

    :param path2single_dirs_json: Path to the directory with the single-directory json files
    :param path2across_dir_csv: Path to the across-directory CSV file
    :param save_path: Path to save the CSV file, incl. filename and file ending
    :param top_n: Number of top words to extract from each topic in the final across-directory CSV file
    :return: -
    """
    existing_topics = set()

    # Read existing topics from the across-directory CSV if it exists
    if os.path.exists(path2across_dir_csv):
        with open(path2across_dir_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    existing_topics.add(row[0])  # Store existing topic IDs

    aggregated_topics = {}

    # Process each JSON file in the directory
    for file_path in get_files(path2single_dirs_json, file_type='json', recursive=True):
        if os.path.isfile(file_path):
            print(f"Processing {file}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            # Extract topics ensuring no duplicates
            for topic_id, words in data.items():
                if words and topic_id not in existing_topics and topic_id not in aggregated_topics:
                    aggregated_topics[topic_id] = ", ".join(words[:top_n])

    # Write to CSV
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Topic ID", "Topic Words"])  # Write header
        for topic_id, words in aggregated_topics.items():
            writer.writerow([topic_id, words])


if __name__ == '__main__':
    local = True
    # single dirs
    input_path = '/Users/klara/Downloads/top-dir-31-01-25/'
    for file in get_files(input_path, file_type='json', recursive=True):
        print(file)
        dir_topic_words2csv(dir_path=file, output_file=file.replace('.json', '.csv'), top_n=5)

    # across dirs
    # path2across_dir_csv = '/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/server-across-dir-incidence-matrix-31-01-25.csv'
    # dir_topic_words2csv_across_dirs(path2single_dirs_json=input_path, path2across_dir_csv=path2across_dir_csv,
    #                                 save_path='/Users/klara/Downloads/server-across-dir-topic-words-31-01-25.csv', top_n=5)

#     path = '/Users/klara/Downloads/KDE_Projekt/sample_data_server' if local else '/norgay/bigstore/kgu/data/ETYNTKE/Workshop/'
#     num_successes = 0
#     limit_num_docs = 2
#     paths = get_files(path)[:limit_num_docs]
#     print('PATHS', paths)
#     for path2file in tqdm.tqdm(paths, desc='Extracting text from pdfs'):
#         text, success = extract_text_from_pdf(path2file)
#         # images = pdf2png(path2file, '/Users/klara/Downloads/', page_num=0)
#         # print(type(images[1]))
#
#         print('NEW DOC', text)
#         num_successes += success
#     print(f"Number of successful extractions: {num_successes}/{len(get_files(path)[:limit_num_docs])}")
