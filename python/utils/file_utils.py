import os
import zipfile
import fnmatch

def zipdir(path, ziph, exclude_patterns=[]):
    """
    Zip an entire directory, optionally excluding some subdirectories and files based on patterns.

    :param path: Path to directory to zip
    :param ziph: ZipFile handle
    :param exclude_patterns: List of patterns to exclude
    """
    for root, dirs, files in os.walk(path):
        # Exclude directories that match any of the exclude_patterns
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)]
        for file in files:
            if not any(fnmatch.fnmatch(file, pattern) for pattern in exclude_patterns):
                ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))

def create_zip(source_directory, output_filename, exclude=[]):
    """
    Create a zip file from a directory, excluding specified patterns.

    :param source_directory: Directory to zip
    :param output_filename: Name of the output zip file
    :param exclude: List of patterns to exclude
    """
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(source_directory, zipf, exclude_patterns=exclude)
