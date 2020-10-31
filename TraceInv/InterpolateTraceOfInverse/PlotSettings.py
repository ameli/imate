# =============
# Plot Settings
# =============

import matplotlib
# matplotlib.use('Agg')
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,InsetPosition,mark_inset
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter,NullFormatter,FormatStrFormatter
import matplotlib.pyplot as plt

# Color palette
import seaborn as sns
# sns.set()

# Axes font size
# sns.set(font_scale=1)
sns.set(font_scale=1.2)

# LaTeX (two font options)
UseLaTeX = True
if UseLaTeX:
    try:
        # plt.rc('text',usetex=True)
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        matplotlib.font_manager._rebuild()
    except:
        raise ValueError('LaTeX is not installed or not configured. Disable "UseLaTeX".')

# Style sheet
sns.set_style("white")
sns.set_style("ticks")

# Font (Note: this should be AFTER the plt.style.use)
plt.rc('font', family='serif')
plt.rcParams['svg.fonttype'] = 'none'  # text in svg file will be text not path.

#from cycler import cycler
#matplotlib.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
