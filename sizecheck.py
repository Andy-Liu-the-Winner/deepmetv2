import os

def check_file_sizes(directory, lower_bound):
    total = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath)
            total += file_size
            if file_size < lower_bound:
                print(file_size)
                print(f"{filename} is smaller than the lower bound.")
    print(f"Total size of files in {directory} is {total} bytes.")
    print(f"Average file size is {total / len(os.listdir(directory))} bytes.")

# Example usage
directory_path = '/hildafs/projects/phy230010p/fep/DeepMETv2/data_znunu/training/processed'
lower_bound = 3200  # Specify the lower bound in bytes

check_file_sizes(directory_path, lower_bound)