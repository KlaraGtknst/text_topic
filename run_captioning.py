import tqdm
import data.files as files


if __name__ == '__main__':
    local = False
    path = '/Users/klara/Downloads/KDE_Projekt/sample_data_server' if local else '/norgay/bigstore/kgu/data/ETYNTKE/Workshop/'
    num_successes = 0
    limit_num_docs = 2
    paths = files.get_files(path)
    print('PATHS', paths)
    for path2file in tqdm.tqdm(paths, desc='Extracting text from pdfs'):
        text, success = files.extract_text_from_pdf(path2file)

        print('NEW DOC', text)
        num_successes += success
    print(f"Number of successful extractions: {num_successes}/{len(files.get_files(path))}")