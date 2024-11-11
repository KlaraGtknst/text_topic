import glob
import pypdf as pdf
import hashlib
import warnings
import tqdm

def get_files(path: str = "/", file_ending: str = "txt"):
    if not path.endswith("/"):
        path += "/"
    return [path for path in glob.glob(f"{path}/**", recursive=True) if path.endswith('pdf')]

def extract_text_from_pdf(path: str):
    # creating a pdf reader object
    try:
        reader = pdf.PdfReader(path)

        text = [page.extract_text() for page in reader.pages]
    except pdf.errors.PdfStreamError as e:
        text = [str(e)]
    return text

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
    with open(path, 'rb') as f: # Open the file to read it's bytes, automatically closes file at end
        fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above
        while len(fb) > 0: # While there is still data being read from the file
            file_hash.update(fb) # Update the hash
            fb = f.read(BLOCK_SIZE)
    id = file_hash.hexdigest()
    return id


def save_sentences_to_file(sentences, dataset_path):
    # save the sentences to a file
    # files.save_text_to_file(sentences, dataset_path + "sentences_old.txt")
    with open(dataset_path + 'sentences1.txt', 'w') as f:
        for i in tqdm(range(len(sentences)), desc='Writing sentences to file'):
            sentence = sentences[i].encode("utf-8", errors="ignore")
            try:
                f.write(f"NEWFILE{sentence}")
            except AttributeError as e:
                # f.write(f"{sentence}\n")
                print(f"Error with sentence {i} encountered: {e}")
                pass
    f.close()

def load_sentences_from_file(dataset_path):
    # load the sentences from a file
    with open(dataset_path + 'sentences1.txt') as f:
        sentences = f.read()  # f.readlines()
    print("File content read successfully")  # Check if this prints
    return sentences