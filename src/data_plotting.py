import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def histogram_plotting(df:pd.DataFrame, wer_columns_list:list, plot_title:str) -> plt.figure:
    # Plot histograms for WER columns without outliers
    plt.figure(figsize=(12, 6))
    for column in wer_columns_list:
        plt.hist(df[column], bins=20, alpha=0.5, label=column)

    #'Distribution of Word Error Rate (WER) without 5% outliers'
    plt.title(plot_title)
    plt.xlabel('WER (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


def boxplot_plotting(df:pd.DataFrame, plot_title:str) -> plt.figure:
    # Plot box plots for WER columns without outliers
    plt.figure(figsize=(12, 6))
    df.boxplot()
    #'Boxplot of Word Error Rate (WER) without 5% outliers'
    plt.title(plot_title)
    plt.ylabel('WER (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def kde_plotting(df:pd.DataFrame, wer_columns_list:list, plot_title:str) -> plt.figure:
    # Plot KDE plot for WER columns without outliers
    plt.figure(figsize=(12, 6))
    for column in wer_columns_list:
        sns.kdeplot(df[column], label=column, fill=True)

    plt.title(plot_title)
    plt.xlabel('WER (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    