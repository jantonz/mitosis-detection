import subprocess
import os

# Activate only once. It downloads the .tar.gz file, untars it,
# moves the images to the corresponding folders and removes the tar file
# Download images from MEGA (official MYTHOS ATIPIA link)
# There is a bandwith quota, which is reached after ~4 files, and then it
# needs to wait 2 hours before continuing the download
mega_files = [
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/aJgjFDLQ",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/6VRhSCDa",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/DEgzWDDK",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/CJgzFSTS",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/OYRU2YYa",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/rZQAnJZS",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/WV5B2YpB",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/DVQlwQYY",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/PNQE1ZQT",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/WYAHTRZb",
    "https://mega.nz/folder/2BxGTQRZ#hcMq5iISw8tWhaSLkcCpYQ/file/7EISEJjT",
]

for mega_file in mega_files:
    print(f"Processing {mega_file}")
    subprocess.run(["mega-get", "--ignore-quota-warn", mega_file, "../dataset/raw"])

    # Untar files into corresponding folders
    filename = [f for f in os.listdir("../dataset/raw") if ".tar" in f][0]
    foldername = filename[0:3]

    print(f"Untaring into {os.path.join('../dataset/raw', filename)}")
    subprocess.run(
        [
            "tar",
            "-xzf",
            f"{os.path.join('../dataset/raw', filename)}",
            "-C",
            "../dataset/raw/train",
            "--strip-components=3",
            f"{foldername}/frames/x40",
        ]
    )

    subprocess.run(
        [
            "tar",
            "-xzf",
            f"{os.path.join('../dataset/raw', filename)}",
            "-C",
            "../labels",
            "--strip-components=2",
            f"{foldername}/mitosis",
        ]
    )
    # Remove tar.gz file
    os.remove(f"{os.path.join('../dataset/raw', filename)}")