import pandas as pd

def parse_log_file(filepath):
    # Define sensor and modality ordering
    sensors = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
    modalities = [
        "Ax", "Ay", "Az",
        "Gx", "Gy", "Gz", 
        "Mx", "My", "Mz",
        "Q1", "Q2", "Q3", "Q4",
    ]
    
    # Build column names
    columns = ["Timestamp_s", "Timestamp_us"]
    for sensor in sensors:
        for modality in modalities:
            columns.append(f"{sensor}_{modality}")
    columns.append("ActivityLabel")
    
    # Load log file
    df = pd.read_csv(filepath, sep="\t", header=None, names=columns)
    return df


def extract_back_data(filepath, output_csv):
    df = parse_log_file(filepath)
    
    # Select only BACK sensor data + timestamps + activity label
    back_columns = ["Timestamp_s", "Timestamp_us"] + \
                   [col for col in df.columns if col.startswith("BACK_")] + \
                   ["ActivityLabel"]
    
    df_back = df[back_columns]
    
    # Save to new CSV
    df_back.to_csv(output_csv, index=False)
    print(f"✅ BACK data saved to {output_csv}")


# Example usage
if __name__ == "__main__":
    log_file = "ai_training/dataset/subject2_mutual4.log"     # input log file
    back_csv = "ai_training/dataset/back_data.csv"  # output CSV file
    extract_back_data(log_file, back_csv)
#file path subject2_mutual4.log
import pandas as pd

def parse_log_file(filepath):
    # Define sensor and modality ordering
    sensors = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
    modalities = [
        "Ax", "Ay", "Az",
        "Gx", "Gy", "Gz", 
        "Mx", "My", "Mz"
        "Q1", "Q2", "Q3", "Q4",

    ]
    
    # Build column names
    columns = ["Timestamp_s", "Timestamp_us"]
    for sensor in sensors:
        for modality in modalities:
            columns.append(f"{sensor}_{modality}")
    columns.append("ActivityLabel")
    
    # Load the log file into a DataFrame
    df = pd.read_csv(filepath, sep="\t", header=None, names=columns)
    
    return df