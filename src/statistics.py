import pandas as pd
from jiwer import wer
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize, RegexpTokenizer
import string
from nltk.probability import FreqDist


def preprocess_text(text: str) -> str:
    # Convert text to lowercase and remove punctuation
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def advanced_statistics(data: pd.DataFrame, column_prefix: str, sentences_column: str, proc_outliers: int = 5):
    for column in data.columns:
        if  column.startswith(column_prefix):
            wer_column_name = column.replace(column_prefix, 'Model_SNR')
            data[wer_column_name] = data.apply(
                lambda row: 100 * wer(
                    preprocess_text(row[sentences_column]), 
                    preprocess_text(row[column])
                ), axis=1
            )

    # Descriptive statistics for WER columns
    wer_columns = [col for col in data.columns if col.startswith('Model_SNR_')]
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
    # Initialize the tokenizer to keep only alphanumeric words
    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenize the transcription and ASR model output columns
    df['transcription_tokens'] = df[transcription_column].apply(lambda x: tokenizer.tokenize(x.lower()))
    df['ASR_model_tokens'] = df[model_recognition_column].apply(lambda x: tokenizer.tokenize(x.lower()))

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

    # Flatten the lists of missed and recognized words
    all_missed_words = [word for words in df['stats'].apply(lambda x: x['missed_words']) for word in words]
    all_recognized_words = [word for words in df['stats'].apply(lambda x: x['recognized_words']) for word in words]

    # Calculate frequencies
    missed_words_freq = FreqDist(all_missed_words)
    recognized_words_freq = FreqDist(all_recognized_words)

    # Total word counts
    total_transcription_words = sum(FreqDist([word for words in df['transcription_tokens'] for word in words]).values())
    total_asr_words = sum(FreqDist([word for words in df['ASR_model_tokens'] for word in words]).values())

    # Get most frequent words in each category
    most_missed_words = missed_words_freq.most_common()
    most_recognized_words = recognized_words_freq.most_common()

    # Calculate percentages and filter out overlapping words
    missed_words_percent = [
        (word, count, (count / total_transcription_words) * 100)
        for word, count in most_missed_words
    ]
    recognized_words_percent = [
        (word, count, (count / total_asr_words) * 100)
        for word, count in most_recognized_words
    ]

    # Filter words that are in both lists
    missed_set = set(word for word, _, _ in missed_words_percent)
    recognized_set = set(word for word, _, _ in recognized_words_percent)

    mostly_missed_words = [item for item in missed_words_percent if item[0] not in recognized_set][:3]
    mostly_recognized_words = [item for item in recognized_words_percent if item[0] not in missed_set][:3]

    return {
        'mostly_missed_words': mostly_missed_words,
        'mostly_recognized_words': mostly_recognized_words
    }
