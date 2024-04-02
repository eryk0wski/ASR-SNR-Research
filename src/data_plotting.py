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
    


def plot_model_comparison(data1:pd.DataFrame, data2:pd.DataFrame,stat_name:str,label1:str,label2:str,label_y_name:str):
    # Extract 'Acceptable percentage' values from both dataframes
    data_1_line = data1.loc[stat_name]
    data_2_line = data2.loc[stat_name]
    
    # Convert strings to floats
    #whisper_acceptable = data_1_line.astype(float)
    #wav2wec_acceptable = wav2wec_acceptable.astype(float)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    data_1_line.plot(label=label1, marker='o')
    data_2_line.plot(label=label2, marker='o')
    
    plt.title(f'{stat_name} plot')
    plt.xlabel('SNR value')
    plt.ylabel(label_y_name)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
