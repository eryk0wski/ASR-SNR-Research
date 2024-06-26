import pandas as pd
import random
import os
import datasets
import soundfile as sf
import pyloudnorm as pyln

def creating_random_split_df(data: pd.DataFrame,batch: int) -> pd.DataFrame:    

    # Check if the input is a DataFrame or a dictionary
    if not isinstance(data, (pd.DataFrame, datasets.arrow_dataset.Dataset)):
        raise ValueError("Input must be a DataFrame or a dictionary.")
    
    total_rows = len(data)

    # if size of batch is bigger than database
    if batch <= 0 or batch > total_rows:
        raise ValueError("Incorrect size of batch")
    
    # Drawing random indexes
    random_indices = random.sample(range(total_rows), batch)

    if isinstance(data, pd.DataFrame):
        # Choosing randomly drawn indexes
        df = data.iloc[random_indices].reset_index(drop=True)
    
    else:
        # Creating dataframe with choosen indexes
        df = pd.DataFrame([data[index] for index in random_indices])

        # Reseting the indexes of df
        df.reset_index(drop=True, inplace=True)
    
    return df


def create_distribution_dict(data:pd.Series,percent:bool = True) -> dict:

    # Raise error if input is not DataFrame
    if not isinstance(data, pd.Series):
        raise ValueError("Input can only be dataframe column")
    
    if percent == True:
        normalised_values = round(data.value_counts() / len(data),3)
    else:
        normalised_values = data.value_counts()

    datasets_dict = normalised_values.to_dict()
    return datasets_dict 



def copy_files(dataframe,df_column_name,destination_folder) -> None:


    # Check if the input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    
    # Check if the input is a string
    if not isinstance(df_column_name, str):
        raise ValueError("Input must be a string.")
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Iterate through the paths in the DataFrame and copy files
    for index, row in dataframe.iterrows():
        source_path = row[df_column_name]
        file_name = os.path.basename(source_path)
        destination_path = os.path.join(destination_folder, file_name)

        # Copy files unless exception

        try:
            shutil.copy2(source_path, destination_path)
            print(f"File {file_name} copied successfully.")
        except Exception as e:
            print(f"Error copying file {file_name}: {e}")




# Function to extract information from the file name
def extract_info(file_name):
    fsID, classID, occurrenceID, sliceID = file_name.replace('.wav', '').split('-')
    return fsID, classID, occurrenceID, sliceID

def noise_dataframe(folder_path) -> pd.DataFrame:
    data = []

    # Loop through each folder from fold1 to fold10
    for i in range(1, 11):
        current_folder = f"fold{i}"
        current_folder_path = os.path.join(folder_path, current_folder)

        # Loop through each file in the current folder
        for file_name in os.listdir(current_folder_path):
            file_path = os.path.join(current_folder_path, file_name)
            fsID, classID, occurrenceID, sliceID = extract_info(file_name)

            # Append data to the list
            data.append({
                'name': file_name,
                'path': file_path,
                'fsID': fsID,
                'classID': classID,
                'occurrenceID': occurrenceID,
                'sliceID': sliceID
            })

    # Create a DataFrame from the list of dictionaries
    return  pd.DataFrame(data)



def audio_normalizer(input_path, output_folder_path,output_file_name, loudness_value):

    # Load audio
    data, rate = sf.read(input_path)

    # Measure the loudness
    meter = pyln.Meter(rate, block_size =0.100)
    loudness = meter.integrated_loudness(data)

    # Loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, loudness_value)

    # Specify the output file paths
    loudness_normalized_output_path = os.path.join(output_folder_path, output_file_name)

    # Write normalized audio to files
    sf.write(loudness_normalized_output_path, loudness_normalized_audio, rate)
