import os


def exists_or_create(path):
    """
    This function checks if a path exists and creates it if it does not.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_or_not(plt, file_name: str, save_path: str, format: str = 'png'):
    """
    This function checks if the user wants to save the plot or not.
    """
    if save_path is not None:
        exists_or_create(path=save_path)
        if not save_path.endswith('/'):
            save_path += '/'
        if not file_name.endswith(format):
            file_name = file_name.split('.')[0] + '.' + format
        plt.savefig(save_path + file_name, bbox_inches='tight', format=format)


def scan_recurse(base_directory: str):
    base_directory = base_directory.split('*')[0] if '*' in base_directory else base_directory

    for entry in os.scandir(base_directory):
        if entry.is_file():
            yield os.path.join(base_directory, entry.name)
        else:  # recurse needs from, otherwise generator object is returned
            yield from scan_recurse(entry.path + '/')
