import os


def get_local_file_size_in_byte(file_path):
    if os.path.islink(file_path):
        file_path = os.path.realpath(file_path)

    size = os.path.getsize(file_path)
    return size
