import os

def rename_folders(directory):
  """
  Renames folders in a directory numerically based on their ascending order.

  Args:
    directory: Path to the directory containing the folders.
  """
  counter = 0
  for filename in sorted(os.listdir(directory)):
    if os.path.isdir(os.path.join(directory, filename)):
      # Construct new filename with padding for consistent formatting
      new_filename = f"{counter:03d}"
      os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
      counter += 1

# Example usage
target_directory = "/home/mohamed/repos/3d_Vision/dl_challenge"
rename_folders(target_directory)

print(f"Folders renamed in directory: {target_directory}")