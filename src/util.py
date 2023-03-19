import os


def getprojdir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
