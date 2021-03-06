import cv2
import os
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import subprocess


def image_show_console(image):
    cv2.imshow("Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_jupyter(image):
    # Convert BGR (cv2) into RGB (matplotlib)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Show the actual image
    plt.imshow(rgb_img)
    plt.show()


def smooth(img):
    dest = cv2.medianBlur(img, 7)
    return dest


def process(path, img):
    image = cv2.imread(os.path.join(path, img), 1)
    image = smooth(image)
    return image


def kmeans(img, name, input_path, output_path, show=None):
    image = img.reshape(img.shape[0] * img.shape[1], 3)
    image = np.float32(image)
    nclusters = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        image, nclusters, None, criteria, attempts, flags
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((img.shape))
    # res2 = img
    d = []
    data = csv.reader(open("../labels/" + name[:-5] + "_mitosis.csv", "r"))
    for j in data:
        d.append([j[0], j[1]])
    e = []
    data = csv.reader(open("../labels/" + name[:-5] + "_not_mitosis.csv", "r"))
    for j in data:
        e.append([j[0], j[1]])
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 250
    params.maxArea = 2500
    params.filterByConvexity = False
    params.minConvexity = 0.95
    params.filterByCircularity = True
    params.minCircularity = 0.65
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(res2)
    maxd = 0
    for i in range(0, len(keypoints)):
        x = int(keypoints[i].pt[0])
        y = int(keypoints[i].pt[1])
        dm = int(keypoints[i].size)
        maxd = max(maxd, dm)
        # Detect mitosis
        for j in range(0, len(d)):
            if (
                x >= float(d[j][0]) - 200
                and x <= float(d[j][0]) + 200
                and y >= float(d[j][1]) - 200
                and y <= float(d[j][1]) + 200
            ):
                try:
                    cv2.imwrite(
                        f"{os.path.join(output_path, '1', name[:-5])}_{i}_true.png",
                        cv2.resize(
                            res2[
                                max(0, y - dm) : max(0, y + dm),
                                max(0, x - dm) : max(0, x + dm),
                            ],
                            (70, 70),
                        ),
                    )
                except:
                    breakpoint()
                d.pop(j)
                break
        # Detect non-mitosis
        for j in range(0, len(e)):
            if (
                x >= float(e[j][0]) - 200
                and x <= float(e[j][0]) + 200
                and y >= float(e[j][1]) - 200
                and y <= float(e[j][1]) + 200
            ):
                try:
                    cv2.imwrite(
                        f"{os.path.join(output_path, '0', name[:-5])}_{i}_false.png",
                        cv2.resize(
                            res2[
                                max(0, y - dm) : max(0, y + dm),
                                max(0, x - dm) : max(0, x + dm),
                            ],
                            (70, 70),
                        ),
                    )
                except:
                    breakpoint()
                e.pop(j)
                break
        # Detect other cells/artifacts
        if (
            not os.path.exists(
                f"{os.path.join(output_path, '1', name[:-5])}_{i}_true.png"
            )
        ) and (
            not os.path.exists(
                f"{os.path.join(output_path, '0', name[:-5])}_{i}_false.png"
            )
        ):
            cv2.imwrite(
                f"{os.path.join(output_path, '0', name[:-5])}_{i}_other.png",
                cv2.resize(
                    res2[
                        max(0, y - dm) : max(0, y + dm),
                        max(0, x - dm) : max(0, x + dm),
                    ],
                    (70, 70),
                ),
            )

    if show in ["jupyter", "console"]:
        im_with_keypoints = cv2.drawKeypoints(
            res2,  # empty: np.zeros(res2.shape, np.uint8) instead of res2
            keypoints,
            np.array([]),
            (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        if show == "console":
            image_show_console(im_with_keypoints)
        if show == "jupyter":
            show_image_jupyter(im_with_keypoints)

    if show in ["jupyter_bw", "console_bw"]:
        imgray_with_keypoints = cv2.drawKeypoints(
            cv2.cvtColor(
                res2, cv2.COLOR_BGR2GRAY
            ),  # empty: np.zeros(res2.shape, np.uint8) instead of res2
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        if show == "console":
            image_show_console(imgray_with_keypoints)
        if show == "jupyter":
            show_image_jupyter(imgray_with_keypoints)


def preprocess(input_path, output_path):
    images = []
    j = 0
    print("Median Blur")
    for i in os.listdir(input_path):
        images.append(process(input_path, i))
        images[j] = kmeans(images[j], i, input_path, output_path, show=None)
        j += 1


# Activate only once. It downloads the .tar.gz file, untars it,
# moves the images to the corresponding folders and removes the tar file
if False:
    # Download images from MEGA (official MYTHOS ATIPIA link)
    # There is a bandwith quota, which is reached after ~4 files, and then it
    # needs to wait 2 houres before continuing the download
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


# Preprocess train set
input_path = "../dataset/raw/train"
output_path = "../dataset/train"

print("Preprocess train set")
preprocess(input_path=input_path, output_path=output_path)


# Preprocess test set
input_path = "../dataset/raw/test"
output_path = "../dataset/test"

print("Preprocess test set")
preprocess(input_path=input_path, output_path=output_path)
