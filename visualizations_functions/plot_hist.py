import matplotlib.pyplot as plt

def plot_hist(arr: list, save_path: str):
    """
    Function to plot a histogram of a given array and save it to a specified path.

    Parameters
    ----------
    arr : list
        The array to plot a histogram of.
    title : str
        The title of the histogram.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    save_path : str
        The path to save the histogram plot to.
    """

    

    plt.hist(arr, bins=20)
    plt.title("Histogram of the array")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    # plt.savefig(save_path)
    plt.show()