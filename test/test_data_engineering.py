import pytest
import pandas as pd
from src.data_engineering import creating_random_split_df, copy_files, create_distribution_dict
from datasets import Dataset 
from tempfile import TemporaryDirectory



# test creating_random_split_df

@pytest.fixture
def example_df():
    return  pd.DataFrame({'col1' : [1,2,3,4,5,6],
                          'col2' : ['a', 'b', 'c', 'd', 'e', 'f'],
                          'col3' : [8,9,10,11,12,13]})


@pytest.fixture
def example_dict():
    dict = {'a': [1, 2, 3, 4, 5],
            'b': [6, 7, 8, 9, 10],
            'c': [11, 12, 13, 14, 15]}
    return Dataset.from_dict(dict)

    
# Checking error raised when putting wrong batch values
def test_creating_split_batch_dimension(example_df):
    with pytest.raises(ValueError):
        k = creating_random_split_df(example_df, len(example_df)+1)

    with pytest.raises(ValueError):
        k = creating_random_split_df(example_df,0)


# Checking error raises when inputing wrong file type
def test_creating_split_wrong_type():
    list = [5,6,3,4,5,6,]

    with pytest.raises(ValueError):
        creating_random_split_df(list,5)
    
    with pytest.raises(ValueError):
        creating_random_split_df('random_string',5)

    with pytest.raises(ValueError):
        creating_random_split_df(15,5)    


#checking length and result datatype when entering dataframe
def test_creating_split_input_dataframe(example_df):
    batch_size = 2
    result_df = creating_random_split_df(example_df, batch_size)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == batch_size

#checking length and result datatype when entering dictionary
def test_creating_split_input_dictionary(example_dict):
    batch_size = 3
    result_df = creating_random_split_df(example_dict, batch_size)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == batch_size

# Checking if all indexes from result occur in sample
def test_output_indexes(example_df):
    batch_size = 2
    result_df = creating_random_split_df(example_df, batch_size)
    assert all(index in example_df.index for index in result_df.index)

#test create_distribution dictionary

#checking error risen while inputting dataframe instead of column
def test_create_distribution_dict_input(example_df):
    with pytest.raises(ValueError):
        create_distribution_dict(example_df)

#checking if output is dictionary and values are rounded
def test_create_distribution_dict_output(example_df):
    output_dict = create_distribution_dict(example_df['col2'])
    assert output_dict == {'a':0.167, 'b':0.167, 'c':0.167, 'd': 0.167, 'e':0.167, 'f':0.167}



'''
@pytest.fixture

def create_temp_folder():
    with TemporaryDirectory() as temp_folder:
        yield temp_folder


def test_copy_files(create_temp_folder):
    temp_folder = create_temp_folder

    # Create a sample DataFrame
    data = {'path': ['/path/to/file1.txt', '/path/to/file2.txt']}
    df = pd.DataFrame(data)

    # Call the function with the sample DataFrame and destination folder
    copy_files(df, 'path', temp_folder)

    # Assert that the files were copied successfully
    assert os.path.exists(os.path.join(temp_folder, 'file1.txt'))
    assert os.path.exists(os.path.join(temp_folder, 'file2.txt'))

def test_copy_files_nonexistent_folder():
    # Create a sample DataFrame
    data = {'path': ['/path/to/nonexistent/file.txt']}
    df = pd.DataFrame(data)

    # Call the function with the sample DataFrame and a nonexistent destination folder
    with pytest.raises(FileNotFoundError):
        copy_files(df, 'path', '/nonexistent/destination/folder')

def test_copy_files_exception():
    # Create a sample DataFrame
    data = {'path': ['/path/to/nonexistent/file.txt']}
    df = pd.DataFrame(data)

    # Create a temporary destination folder
    with TemporaryDirectory() as temp_folder:
        # Call the function with the sample DataFrame and destination folder
        with pytest.raises(Exception):
            copy_files(df, 'path', temp_folder)'''