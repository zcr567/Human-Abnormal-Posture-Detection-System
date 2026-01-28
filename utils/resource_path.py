import sys
from os.path import abspath


def resource_path(relative_path) -> str:
    """Get the packaged resource directory. When building the .exe file, use this function to find the real file path.
    :param relative_path: relative path in the development environment,a string or a pathlike object.
    :return absolute path of the packaged resource."""
    if hasattr(sys, '_MEIPASS'):
        # packaged env
        # noinspection PyProtectedMember
        base_path = str(sys._MEIPASS)
    else:
        # development env
        base_path = abspath("..")
    return base_path + '/' + relative_path
