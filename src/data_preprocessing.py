import pandas as pd

def load_data(file_path):
  
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        return None

if __name__ == "__main__":
  
    file_path = r"....path\tomato-quality production\data\sample_data.csv"
    data = load_data(file_path)
    if data is not None:
        print(data.head())
