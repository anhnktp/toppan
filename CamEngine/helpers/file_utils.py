import os

def remove_file(fname):
    try:
        os.remove(fname)
    except OSError:
        pass