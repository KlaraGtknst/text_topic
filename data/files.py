import glob
import pypdf as pdf
import hashlib
import warnings

def get_files(path: str = "/", file_ending: str = "txt"):
    if not path.endswith("/"):
        path += "/"
    return glob.glob(f"{path}*.{file_ending}")

def extract_text_from_pdf(path: str):
    # creating a pdf reader object
    reader = pdf.PdfReader(path)

    text = [page.extract_text() for page in reader.pages]
    return text

def pdf_to_str(path: str) -> str:
    '''
    :param path: path to pdf file
    :return: text from pdf file

    This function extracts the text from a pdf file.
    cf. https://pypi.org/project/PyPDF2/
    '''
    with warnings.catch_warnings(action="ignore"):
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
