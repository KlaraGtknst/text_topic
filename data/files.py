import glob
import pypdf as pdf
import hashlib
import warnings
import tqdm
import logging
import utils.os_manipulation as osm

# Suppress logging from pypdf
logging.getLogger("pypdf").setLevel(logging.CRITICAL)

def get_files(path: str = "/"):
    """
    This function returns a list of all file paths that end with 'pdf' in a directory.
    :param path: Path to the directory; if no path is given, the function returns all pdf files in the current directory.
    :return: List of file paths
    """
    if not path.endswith("/"):
        path += "/"
    return [path for path in glob.glob(f"{path}/**", recursive=True) if path.endswith('pdf')]

def extract_text_from_pdf(path: str):
    """
    This function extracts the text from a pdf file.
    If the pdf file is not readable, the function returns a list which contains the error message.
    :param path: Path to the pdf file
    :return: List of text from the pdf file; each entry is the text of one page
    """
    # creating a pdf reader object
    try:
        reader = pdf.PdfReader(path)

        text = []
        for page in reader.pages:
            try:
                # Attempt to extract text
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                else:
                    text.append("Empty page.")
            except Exception as e:
                text.append(f"Error extracting text from page: {e}")
        return "\n".join(text), True
    except pdf.errors.PdfStreamError as e:
        return [str(e)], False
    except AttributeError as e:     # Document is encrypted
        return [str(e)], False
    except ValueError as e:         # negative seek value -1
        return [str(e)], False
    except Exception as e:  # all other errors
        return [str(e)], False

def pdf_to_str(path: str) -> str:
    '''
    :param path: path to pdf file
    :return: text from pdf file

    This function extracts the text from a pdf file.
    cf. https://pypi.org/project/PyPDF2/
    '''
    #warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
    with warnings.catch_warnings(action="ignore"):
        warnings.simplefilter("ignore")  # Ignore all warnings
        try:
            reader = pdf.PdfReader(path)

            text = ''
            for page in reader.pages:
                text += page.extract_text()

            return text
        except:
            # TODO: fix missing EOF marker in pdf
            return ''

def get_hash_file(path: str):
    '''
    :param path: path to the file
    :return: hash of the file
    '''
    BLOCK_SIZE = 65536000 # The size of each read from the file
    file_hash = hashlib.sha256() # Create the hash object, can use something other than `.sha256()` if you wish
    with open(path, 'rb') as f: # Open the file to read its bytes, automatically closes file at end
        fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above
        while len(fb) > 0: # While there is still data being read from the file
            file_hash.update(fb) # Update the hash
            fb = f.read(BLOCK_SIZE)
    id = file_hash.hexdigest()
    return id


def save_sentences_to_file(sentences, dataset_path, save_filename:str='sentences2.txt'):
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

def load_sentences_from_file(dataset_path, filename:str='sentences1.txt'):
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
    df.to_csv(path + file_name + '.csv', index=True)
    print(f"Dataframe saved to {path}")



# if __name__ == '__main__':
#     path = '/Users/klara/Downloads/Exotic Weapons'
#     num_successes = 0
#     paths = get_files(path)[:10]
#     for i in tqdm.tqdm(range(len(paths)), desc='Extracting text from pdfs'):
#         path2file = paths[i]
#         text, success = extract_text_from_pdf(path2file)
#         print(text)
#         num_successes += success
#     print(f"Number of successful extractions: {num_successes}/{len(get_files(path))}")