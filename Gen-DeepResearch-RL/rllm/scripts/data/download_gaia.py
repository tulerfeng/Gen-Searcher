import os

import gdown

import rllm

# Define the Google Drive file IDs for the Gaia files
FILE_IDS = {
    "gaia.json": "1cVW-HPyboCcwRrn3Gtsuay8jTLxcvrWH",
    "gaia_files.zip": "1rsU2xs21f3Lmn0z0zCxXoKxXkfUodFWK",
}

# Get the rllm package path
RLLM_PATH = os.path.dirname(os.path.dirname(rllm.__file__))

# Define the destination paths

WEB_PATH = os.path.join(RLLM_PATH, "rllm/data/train/web")


def download_and_extract_files():
    # Create the necessary directories
    os.makedirs(WEB_PATH, exist_ok=True)

    # Download the JSON file
    print("Downloading gaia.json...")
    output_path = os.path.join(WEB_PATH, "gaia.json")
    gdown.download(f"https://drive.google.com/uc?id={FILE_IDS['gaia.json']}", output_path, quiet=False)

    # Download and extract the Gaia files
    print("Downloading and extracting Gaia files...")
    temp_zip_path = "./gaia_files.zip"
    gdown.download(f"https://drive.google.com/uc?id={FILE_IDS['gaia_files.zip']}", temp_zip_path, quiet=False)

    # Extract the zip file
    import zipfile

    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        zip_ref.extractall(WEB_PATH)

    # Remove the temporary zip file
    os.remove(temp_zip_path)

    print("All Gaia files downloaded and extracted successfully.")


if __name__ == "__main__":
    download_and_extract_files()
