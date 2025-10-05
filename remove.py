import os
import pandas as pd
import glob

def cleanup_small_csv_files(data_dir='dataset', min_rows=7):
    """
    Delete CSV files that have fewer than min_rows data rows
    
    Args:
        data_dir: Directory to search for CSV files
        min_rows: Minimum number of data rows required (default: 7)
    """
    print("=" * 70)
    print("CSV FILE CLEANUP - DELETE FILES WITH TOO FEW ROWS")
    print("=" * 70)
    print(f"\nSearching in: {os.path.abspath(data_dir)}")
    print(f"Minimum rows required: {min_rows}")
    print("\n" + "-" * 70)
    
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"\n⚠ No CSV files found in '{data_dir}' or its subfolders!")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s)\n")
    
    # Track statistics
    files_deleted = 0
    files_kept = 0
    files_error = 0
    deleted_files_list = []
    
    # Process each file
    for csv_file in csv_files:
        rel_path = os.path.relpath(csv_file, data_dir)
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            num_rows = len(df)
            
            # Check if file should be deleted
            if num_rows < min_rows:
                print(f"🗑 DELETING: {rel_path}")
                print(f"   Rows: {num_rows} (< {min_rows})")
                
                os.remove(csv_file)
                files_deleted += 1
                deleted_files_list.append((rel_path, num_rows))
            else:
                print(f"✓ KEEPING: {rel_path}")
                print(f"   Rows: {num_rows}")
                files_kept += 1
                
        except Exception as e:
            print(f"✗ ERROR: {rel_path}")
            print(f"   Error: {str(e)}")
            files_error += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(csv_files)}")
    print(f"  ✓ Files kept: {files_kept}")
    print(f"  🗑 Files deleted: {files_deleted}")
    if files_error > 0:
        print(f"  ✗ Files with errors: {files_error}")
    
    # List deleted files
    if deleted_files_list:
        print("\n" + "-" * 70)
        print("DELETED FILES:")
        print("-" * 70)
        for filepath, rows in deleted_files_list:
            print(f"  - {filepath} ({rows} rows)")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    # Run cleanup with default settings
    cleanup_small_csv_files('dataset', min_rows=7)
    
    # Optional: Show remaining files count by type
    print("\n" + "=" * 70)
    print("REMAINING FILES BY ACTIVITY TYPE")
    print("=" * 70)
    
    csv_files = glob.glob(os.path.join('dataset', '**', '*.csv'), recursive=True)
    
    activity_counts = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Activity_Type' in df.columns and len(df) > 0:
                activity = df['Activity_Type'].iloc[0]
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
        except:
            pass
    
    if activity_counts:
        print("\nFiles by activity type:")
        for activity, count in sorted(activity_counts.items()):
            print(f"  {activity}: {count} files")
    else:
        print("\nNo valid files remaining or no Activity_Type column found.")