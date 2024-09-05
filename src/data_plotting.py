import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def histogram_plotting(df:pd.DataFrame, wer_columns_list:list, plot_title:str) -> plt.figure:
    # Plot histograms for WER columns without outliers
    plt.figure(figsize=(12, 6))
    for column in wer_columns_list:
        plt.hist(df[column], bins=24, alpha=0.5, label=column)

    #'Distribution of Word Error Rate (WER) without 5% outliers'
    plt.title(plot_title)
    plt.xlabel('WER')
    plt.ylabel('Częstotliwość')
    plt.legend()
    plt.grid(True)
    plt.show()


def boxplot_plotting(df: pd.DataFrame, plot_title: str) -> plt.figure:
    # Plot box plots for WER columns without outliers
    plt.figure(figsize=(12, 6))
    df.boxplot()

    # Set the plot title and y-axis label
    plt.title(plot_title)
    plt.ylabel('WER')
    plt.xlabel('SNR')
    
    # Set custom x-axis ticks with labels
    x_ticks = [1, 2, 3, 4, 5, 6]  # Dummy positions for each label
    x_labels = ['100', '50', '25', '10', '5', '0.1']
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45)
    
    # Add grid lines
    plt.grid(True)
    
    # Show the plot
    plt.show()

def kde_plotting(df:pd.DataFrame, wer_columns_list:list, plot_title:str) -> plt.figure:
    # Plot KDE plot for WER columns without outliers
    plt.figure(figsize=(12, 6))
    for column in wer_columns_list:
        sns.kdeplot(df[column], label=column, fill=True)

    plt.title(plot_title)
    plt.xlabel('WER')
    plt.ylabel('Gęstość')
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)
    plt.show()
    


def plot_model_comparison(data1:pd.DataFrame, data2:pd.DataFrame,stat_name:str,label1:str,label2:str,label_y_name:str, plot_title, inverted = False):

    #rozwiazanie jest idiotyczne ale byly tutaj problemy natury dziwnej

    data_1_line = []

    data_2_line = []

    data_1_line_temp = data1.loc[stat_name]

    for i in range(len(data_1_line_temp)):
        data_1_line.append(data_1_line_temp[i])

    data_2_line_temp = data2.loc[stat_name]

    for i in range(len(data_2_line_temp)):
        data_2_line.append(data_2_line_temp[i])


    print(data_1_line)
    print(data_2_line)
    # Define the specific x-axis tick values you want to display
    xticks_values = ['100', '50', '25', '10', '5', '0.1']

    print(xticks_values)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot data with specified x-axis values, assuming x-values are indices
    plt.plot(xticks_values, data_1_line, label=label1, marker='o')
    plt.plot(xticks_values, data_2_line, label=label2, marker='o')

    # Set plot details
    plt.title(plot_title)
    plt.xlabel('SNR')
    plt.ylabel(label_y_name)
    plt.legend()
    plt.grid(True)

    # Set the x-axis ticks to the specified values
    plt.xticks(rotation=45)

    if inverted == True:
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()