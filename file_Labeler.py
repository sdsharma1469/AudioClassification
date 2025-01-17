import os

def rename_files(directory):
    # Traverse through each subdirectory (siren, baby, etc.)
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        
        # Check if it is a directory
        if os.path.isdir(subdir_path):
            # List all files in the subdirectory
            files = os.listdir(subdir_path)
            # Initialize counter
            counter = 1

            # Loop through files and rename them
            for file in files:
                # Get the file extension
                file_ext = file.split('.')[-1].lower()

                # If it's a file (not a folder) and is not hidden
                if os.path.isfile(os.path.join(subdir_path, file)) and not file.startswith('.'):
                    # Generate the new filename
                    new_file_name = f"{subdir}_{str(counter).zfill(3)}.{file_ext}"
                    # Get the old and new file paths
                    old_file_path = os.path.join(subdir_path, file)
                    new_file_path = os.path.join(subdir_path, new_file_name)

                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {file} -> {new_file_name}")

                    # Increment the counter
                    counter += 1

# Example usage
directory = 'test'  # Set the root directory to start from
rename_files(directory)
