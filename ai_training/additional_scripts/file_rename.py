import os
import pandas as pd
import glob
from collections import Counter

def get_dominant_activity(file_path):
    """
    Get the most common activity type from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Most common activity type or None if error
    """
    try:
        df = pd.read_csv(file_path)
        
        if 'Activity_Type' not in df.columns:
            print(f"  ⚠ No 'Activity_Type' column in {file_path}")
            return None
        
        # Get the most common activity
        activity_counts = df['Activity_Type'].value_counts()
        dominant_activity = activity_counts.index[0]
        
        # Show if file has mixed activities
        if len(activity_counts) > 1:
            print(f"  ℹ Multiple activities found: {dict(activity_counts)}")
            print(f"    Using dominant: '{dominant_activity}'")
        
        return dominant_activity
        
    except Exception as e:
        print(f"  ✗ Error reading {file_path}: {str(e)}")
        return None

def rename_file_by_activity(file_path):
    """
    Rename a CSV file to include its activity type
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if renamed, False otherwise
    """
    # Get dominant activity
    activity = get_dominant_activity(file_path)
    
    if not activity:
        return False
    
    # Parse current filename
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    name_without_ext, ext = os.path.splitext(filename)
    
    # Check if activity is already in filename
    if activity.lower() in filename.lower():
        print(f"  - Already contains '{activity}': {filename}")
        return False
    
    # Parse the filename structure: data_P{id}_{device}_{old_activity}_{timestamp}.csv
    parts = name_without_ext.split('_')
    
    # Build new filename
    if len(parts) >= 4 and parts[0] == 'data':
        # Standard format: data_P1_DEVICE_001_OldActivity_20251005_020345
        participant = parts[1]  # P1
        device = parts[2]       # DEVICE
        device_num = parts[3] if len(parts) > 3 else ''  # 001
        
        # Get timestamp (last parts that look like date/time)
        timestamp_parts = []
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit() and len(parts[i]) >= 6:
                timestamp_parts.insert(0, parts[i])
            else:
                break
        
        timestamp = '_'.join(timestamp_parts) if timestamp_parts else ''
        
        # Construct new filename
        if timestamp:
            if device_num:
                new_filename = f"data_{participant}_{device}_{device_num}_{activity}_{timestamp}{ext}"
            else:
                new_filename = f"data_{participant}_{device}_{activity}_{timestamp}{ext}"
        else:
            if device_num:
                new_filename = f"data_{participant}_{device}_{device_num}_{activity}{ext}"
            else:
                new_filename = f"data_{participant}_{device}_{activity}{ext}"
    else:
        # Simple format: just append activity before extension
        new_filename = f"{name_without_ext}_{activity}{ext}"
    
    # Create new path
    new_path = os.path.join(directory, new_filename)
    
    # Check if target already exists
    if os.path.exists(new_path):
        print(f"  ⚠ Target file already exists: {new_filename}")
        return False
    
    # Rename the file
    try:
        os.rename(file_path, new_path)
        print(f"  ✓ Renamed:")
        print(f"    From: {filename}")
        print(f"    To:   {new_filename}")
        return True
    except Exception as e:
        print(f"  ✗ Error renaming: {str(e)}")
        return False

def rename_all_csv_files(root_folder='dataset'):
    """
    Rename all CSV files in dataset folder based on their activity type
    
    Args:
        root_folder: Root directory to start searching
    """
    print("=" * 70)
    print("CSV FILE RENAMING SCRIPT - BASED ON ACTIVITY TYPE")
    print("=" * 70)
    print(f"\nRoot folder: {root_folder}\n")
    print("-" * 70)
    
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"\n⚠ No CSV files found in '{root_folder}' or its subfolders!")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s)\n")
    
    # Track statistics
    files_renamed = 0
    files_skipped = 0
    files_error = 0
    
    # Process each file
    for csv_file in csv_files:
        rel_path = os.path.relpath(csv_file, root_folder)
        print(f"\nProcessing: {rel_path}")
        
        result = rename_file_by_activity(csv_file)
        
        if result is True:
            files_renamed += 1
        elif result is False:
            files_skipped += 1
        else:
            files_error += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(csv_files)}")
    print(f"  ✓ Files renamed: {files_renamed}")
    print(f"  - Files skipped: {files_skipped}")
    if files_error > 0:
        print(f"  ✗ Files with errors: {files_error}")
    
    # Show final file list
    print("\n" + "=" * 70)
    print("FINAL FILE LIST")
    print("=" * 70)
    
    new_csv_files = glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True)
    
    # Group by activity
    activity_files = {}
    for csv_file in new_csv_files:
        activity = get_dominant_activity(csv_file)
        if activity:
            if activity not in activity_files:
                activity_files[activity] = []
            activity_files[activity].append(os.path.basename(csv_file))
    
    for activity in sorted(activity_files.keys()):
        print(f"\n{activity}: ({len(activity_files[activity])} files)")
        for filename in sorted(activity_files[activity]):
            print(f"  - {filename}")
    
    print("\nDone!")

if __name__ == "__main__":
    # Run the script
    rename_all_csv_files('dataset')