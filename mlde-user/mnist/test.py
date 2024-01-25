import data
import urllib.parse
import os
import logging
import requests
import shutil

def download_dataset(download_directory: str) -> str:
    url = "https://s3-us-west-2.amazonaws.com/determined-ai-test-data/pytorch_mnist.tar.gz"
    url_path = urllib.parse.urlparse(url).path
    basename = url_path.rsplit("/", 1)[1]

    download_directory = os.path.join(download_directory, "MNIST")
    os.makedirs(download_directory, exist_ok=True)
    filepath = os.path.join(download_directory, basename)
    if not os.path.exists(filepath):
        logging.info("Downloading {} to {}".format(url, filepath))

        r = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    shutil.unpack_archive(filepath, download_directory)

    return os.path.dirname(download_directory)

if __name__ == '__main__':
    dl_directory = download_dataset(download_directory=".")
    train_data = data.get_dataset(dl_directory, train=True)
    print("Done")
