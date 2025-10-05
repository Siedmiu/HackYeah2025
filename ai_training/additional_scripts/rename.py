import os
import pandas as pd
import glob

def rename_activities_in_csv(file_path, replacements):
    """
    Rename activities in a CSV file
    
    Args:
        file_path: Path to the CSV file
        replacements: Dictionary of old_name -> new_name mappings
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if Activity_Type column exists
        if 'Activity_Type' not in df.columns:
            print(f"  ⚠ Skipping {file_path}: No 'Activity_Type' column found")
            return False
        
        # Track if any changes were made
        changes_made = False
        original_values = df['Activity_Type'].value_counts().to_dict()
        
        # Replace values
        for old_name, new_name in replacements.items():
            if old_name in df['Activity_Type'].values:
                df['Activity_Type'] = df['Activity_Type'].replace(old_name, new_name)
                changes_made = True
        
        # Save if changes were made
        if changes_made:
            df.to_csv(file_path, index=False)
            new_values = df['Activity_Type'].value_counts().to_dict()
            
            print(f"  ✓ Updated: {os.path.basename(file_path)}")
            print(f"    Before: {original_values}")
            print(f"    After:  {new_values}")
            return True
        else:
            print(f"  - No changes needed: {os.path.basename(file_path)}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {str(e)}")
        return False

def process_dataset_folder(root_folder='dataset'):
    """
    Process all CSV files in dataset folder and subfolders
    
    Args:
        root_folder: Root directory to start searching
    """
    # Define replacements
    replacements = {
        'Udarzenia_lewo': 'Udarzenia_lewa'
    }
    
    print("=" * 70)
    print("CSV ACTIVITY RENAMING SCRIPT")
    print("=" * 70)
    print(f"\nRoot folder: {root_folder}")
    print(f"Replacements:")
    for old, new in replacements.items():
        print(f"  '{old}' → '{new}'")
    print("\n" + "-" * 70)
    
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"\n⚠ No CSV files found in '{root_folder}' or its subfolders!")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s)\n")
    
    # Process each file
    files_changed = 0
    files_unchanged = 0
    files_error = 0
    
    for csv_file in csv_files:
        # Show relative path for cleaner output
        rel_path = os.path.relpath(csv_file, root_folder)
        print(f"\nProcessing: {rel_path}")
        
        result = rename_activities_in_csv(csv_file, replacements)
        
        if result is True:
            files_changed += 1
        elif result is False:
            files_unchanged += 1
        else:
            files_error += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(csv_files)}")
    print(f"  ✓ Files changed: {files_changed}")
    print(f"  - Files unchanged: {files_unchanged}")
    if files_error > 0:
        print(f"  ✗ Files with errors: {files_error}")
    print("\nDone!")

if __name__ == "__main__":
    # Run the script
    process_dataset_folder('dataset')
    
    # Optional: Verify changes by showing unique activities across all files
    print("\n" + "=" * 70)
    print("VERIFICATION - Unique Activities Across All Files")
    print("=" * 70)
    
    all_activities = set()
    csv_files = glob.glob(os.path.join('dataset', '**', '*.csv'), recursive=True)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Activity_Type' in df.columns:
                all_activities.update(df['Activity_Type'].unique())
        except:
            pass
    
    if all_activities:
        print("\nAll unique activities found:")
        for activity in sorted(all_activities):
            print(f"  - {activity}")
    else:
        print("\nNo activities found or no valid CSV files.")