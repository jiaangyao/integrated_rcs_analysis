import os
import zipfile
import fnmatch
import subprocess

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

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
        return {"commit": commit, "branch": branch}
    except subprocess.CalledProcessError:
        print("An error occurred while trying to fetch git info")
        return None