import pandas as pd
from jiwer import wer
import nltk
from nltk.tokenize import word_tokenize


def advanced_statistics(data:pd.DataFrame,column_prefix:str,sentences_column:str,proc_outliers:int = 5):
    for column in data.columns:
        if column.startswith(column_prefix):
            wer_column_name = column.replace(column_prefix, 'WER_SNR_')
            data[wer_column_name] = data.apply(lambda row: 100 * wer(row[sentences_column], row[column]), axis=1)

    # Descriptive statistics for WER columns
    wer_columns = [col for col in data.columns if col.startswith('WER_SNR_')]
    descriptive_stats = data[wer_columns].describe()

    # Compute mean excluding top selected percent values
    low_proc = proc_outliers/100
    high_proc = (100-proc_outliers)/100
    data[wer_columns] = data[wer_columns].apply(pd.to_numeric, errors='coerce')
    mean_excluding_outliers = data[wer_columns].apply(lambda x: x[(x >= x.quantile(low_proc)) & (x <= x.quantile(high_proc))].mean())
    descriptive_stats.loc['mean without 2%'] = mean_excluding_outliers

    # Count of zero values in each WER column
    zero_counts = (data[wer_columns] == 0).sum()
    descriptive_stats.loc['Perfect outputs'] = zero_counts


    # Count of rows with WER under 10 (Very good)
    count_very_good = ((data[wer_columns] > 0) & (data[wer_columns] < 10)).sum()
    descriptive_stats.loc['Very good outputs'] = count_very_good

    # Count of rows with WER under 20 (Acceptable)
    count_acceptable = ((data[wer_columns] >= 10) & (data[wer_columns] < 20)).sum()
    descriptive_stats.loc['Acceptable outputs'] = count_acceptable

    # Acceptability percentage - number of lines with WER 20 or under.
    count_acceptable_or_less = ((data[wer_columns] <= 20)).sum()
    acceptability_percent = (count_acceptable_or_less / len(data)) * 100
    descriptive_stats.loc['Acceptable percentage'] = acceptability_percent

    # Round the descriptive statistics table
    descriptive_stats = descriptive_stats.round(2)

    # Convert rounded values with .00 to the 
    #Doesn't work, shouldn't work, but I had hope
    for col in descriptive_stats.columns:
        descriptive_stats[col] = descriptive_stats[col].apply(lambda x: int(x) if x == int(x) else x)

    return descriptive_stats


def word_frequency(df, transcription_column, model_recognition_column):
    
    # Tokenize the transcription column
    df['transcription_tokens'] = df[transcription_column].apply(word_tokenize)
    df['ASR_model_tokens'] = df[model_recognition_column].apply(word_tokenize)

    # Compute statistics
    def compute_stats(row):
        trans_tokens = row['transcription_tokens']
        ASR_tokens = row['ASR_model_tokens']
        
        
        missed_words = [word for word in trans_tokens if word not in ASR_tokens]
        
        recognized_words = [word for word in ASR_tokens if word in trans_tokens]
        
        return {
            'missed_words': missed_words,
            'recognized_words': recognized_words
        }

    df['stats'] = df.apply(compute_stats, axis=1)

    # Get mostly missed word
    missed_words_freq = nltk.FreqDist([word for words in df['stats'].apply(lambda x: x['missed_words']) for word in words])
    mostly_missed_word = missed_words_freq.max()

    # Get mostly recognized word
    recognized_words_freq = nltk.FreqDist([word for words in df['stats'].apply(lambda x: x['recognized_words']) for word in words])
    mostly_recognized_word = recognized_words_freq.max()

    return {
        'mostly_missed_word': mostly_missed_word,
        'mostly_recognized_word': mostly_recognized_word
    }
