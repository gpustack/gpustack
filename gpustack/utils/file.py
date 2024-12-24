import os
import shutil

from gpustack.utils import platform


def get_local_file_size_in_byte(file_path):
    if os.path.islink(file_path):
        file_path = os.path.realpath(file_path)

    size = os.path.getsize(file_path)
    return size


def copy_with_owner(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    copy_owner_recursively(src, dst)


def copy_owner_recursively(src, dst):
    if platform.system() in ["linux", "darwin"]:
        st = os.stat(src)
        os.chown(dst, st.st_uid, st.st_gid)
        for dirpath, dirnames, filenames in os.walk(dst):
            for dirname in dirnames:
                os.chown(os.path.join(dirpath, dirname), st.st_uid, st.st_gid)
            for filename in filenames:
                os.chown(os.path.join(dirpath, filename), st.st_uid, st.st_gid)
