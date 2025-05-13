import lzma
import os
import tarfile
from pathlib import Path
 
 
def decompress_xz_files(src_dir, dest_dir, start_index=1, end_index=1000):
    """Decompress .xz files containing multiple documents and copy each document to the destination directory."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
 
    for i in range(start_index, end_index + 1):
        src_file = f"{src_dir}/urlsf_subset01-{i}_data.xz"
        if os.path.exists(src_file):
            # Check if the file is a tarball
            if tarfile.is_tarfile(src_file):
                # Open the tarball file
                with tarfile.open(src_file, mode='r:xz') as tar:
                    tar.extractall(path=dest_dir)
                    print(f"Extracted all contents of {src_file} to {dest_dir}")
            else:
                # Handle regular .xz files
                dest_file_path = os.path.join(dest_dir, f"extracted_content_{i}.txt")
                with lzma.open(src_file, 'rt') as file:
                    content = file.read()
                with open(dest_file_path, 'w') as out_file:
                    out_file.write(content)
                print(f"Decompressed and copied content from {src_file} to {dest_file_path}")
        else:
            print(f"File {src_file} does not exist")


# Specify your source and destination directories
source_directory = '/data2/jun/my-distill/data/openwebtext_origin'
destination_directory = '/data2/jun/my-distill/data/openwebtext_txt'
 
# Call the function
decompress_xz_files(source_directory, destination_directory)