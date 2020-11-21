import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("seaborn")
mpl.rcParams["font.family"] = "serif"


def comp_plot(df, actual, comparison, maturity="maturity_in_years", diff="Price_Diff", title=None):
    ''' Simple function for comparing actuals versus projected '''
    plt.figure(figsize=(20, 12))
    plt.subplot(121)
    plt.scatter(df[maturity], df[actual], marker="+", label="Actual")
    if maturity != comparison:
        plt.scatter(df[maturity], df[comparison], marker="o", label="Calculated")
    plt.legend(loc=0)
    plt.xlabel(maturity)
    plt.ylabel(actual)

    if title:
        if isinstance(title, list):
            plt.title(title[0])
        else:
            plt.title(title)

    if diff in df.columns:
        plt.subplot(122)
        plt.scatter(df[maturity], df[diff])
        if title and isinstance(title, list) and len(title) > 1:
            plt.title(title[1])
        plt.xlabel(maturity)
        plt.ylabel(diff)
