# =======
# Imports
# =======

import os
import matplotlib
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,InsetPosition,mark_inset
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter,NullFormatter,FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from distutils.spawn import find_executable

# Check DISPLAY
if not bool(os.environ.get('DISPLAY',None)):
    # No display found. Using non-interactive Agg backend.
    plt.switch_backend('agg')

# Remove plt.tight_layout() warning
import logging
logging.captureWarnings(True)
import warnings
warnings.filterwarnings(action='ignore',module='matplotlib',category=UserWarning,message=('This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.'))

# ==================
# Load Plot Settings
# ==================

def LoadPlotSettings():
    """
    Specifies general settings for the plots in the example scripts, 
    namely, it sets plot themes by ``seaborn``, fonts by LaTeX if available.
    """

    # Color palette
    import seaborn as sns
    # sns.set()

    # LaTeX
    # if find_executable('latex'):
    #     try:
    #         # plt.rc('text',usetex=True)
    #         matplotlib.rcParams['text.usetex'] = True
    #         matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    #         matplotlib.font_manager._rebuild()
    #
    #         # LaTeX font is a bit small. Increaset axes font size
    #         sns.set(font_scale=1.2)
    #     except:
    #         pass

    # Style sheet
    sns.set_style("white")
    sns.set_style("ticks")

    # Font (Note: this should be AFTER the plt.style.use)
    plt.rc('font',family='serif')
    plt.rcParams['svg.fonttype'] = 'none'  # text in svg file will be text not path.

    #from cycler import cycler
    #matplotlib.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

# =========
# Save Plot
# =========

def SavePlot(plt,Filename,TransparentBackground=True):
    """
    Saves plots.

    :param plt: matplotlib.pyplot object for the plots.
    :type plt: matplotlib.pyplot

    :param Filename: Name of the file without extension or directory name.
    :type Filename: string

    Format:
        The file is saved in both ``svg`` and ``pdf`` format.
    
    Directory:
        The plot is saved in the directory ``/docs/images/`` with respect to the package root,
        if this directory is exists and writable. Otherwise, the plot is saved in the *current*
        directory of the user.
    """

    # Get the root directory of the package (parent directory of this script)
    FileDirectory = os.path.dirname(os.path.realpath(__file__))
    ParentDirectory = os.path.dirname(FileDirectory)
    SecondParentDirectory = os.path.dirname(ParentDirectory)

    # Try to save in the docs/images directory. Check if exists and writable
    SaveDir = os.path.join(SecondParentDirectory,'docs','images')
    if (not os.path.isdir(SaveDir)) or (not os.access(SaveDir,os.W_OK)):

        # Write in the current working directory
        SaveDir = os.getcwd()

    # Save plot in both svg and pdf format
    Filename_SVG = Filename + '.svg'
    if os.access(SaveDir,os.W_OK):
        SaveFullname_SVG = os.path.join(SaveDir,Filename_SVG)
        plt.savefig(SaveFullname_SVG,transparent=TransparentBackground,bbox_inches='tight')
        print('Plot saved to "%s".'%(SaveFullname_SVG))
    else:
        print('Cannot save plot to %s. Directory is not writable.'%SaveDir)
