import pandas as pd
import os

# Change to the parent directory
# os.chdir('..')

def load_data():
    """
    Load the dataset from a specified path and return a DataFrame.
    """
    file_path = '../data/raw/MachineLearningRating_v3.txt'  # Updated path as `os.chdir` already changes the working directory

    try:
        # Load the dataset with delimiter '|'
        df = pd.read_csv(file_path, delimiter='|')
        
        # Display the first few rows of the DataFrame (optional for debugging)
        print("Dataset loaded successfully. Preview:")
        # print(df.head())
        
        return df

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
