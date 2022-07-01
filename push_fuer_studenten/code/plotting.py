import matplotlib.pyplot as plt
import matplotlib as mpl

def set_rcParams():
    
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.titlesize"]= 8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["legend.title_fontsize"] = 7
    return None

def createfig(ncols=1, nrows=1, sizex=6, sizey=4):
    fig = plt.figure(figsize=(sizex, sizey), constrained_layout=True)
    grid = fig.add_gridspec(ncols=ncols, nrows=nrows, hspace=0.2, wspace=0.15)
    
    axs = [0]*ncols*nrows
    for i,ax in enumerate(axs):
        axs[i] = fig.add_subplot(grid[i])
    return axs

def setabc(axs):
    letters = ['a','b','c','d']
    l=len(axs)
    abc = letters[:l]
    for i,lab in enumerate(abc):
        axs[i].text(-.25,1.1,lab, size=12, weight='bold', color='black', transform=axs[i].transAxes)
    return None