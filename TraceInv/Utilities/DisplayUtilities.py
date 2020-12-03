# ===========
# Is Notebook
# ===========

def IsNotebook():
    """
    Returns ``True`` if this script is run in a notebook. Retruns ``False`` otherwise, including both
    ipython and python.
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False      # Probably standard Python interpreter
