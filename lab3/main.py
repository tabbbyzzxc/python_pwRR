import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io, color

Sx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

Sy = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

L = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

Px = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

G = (1 / 16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

W = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


def apply_convolution(image, kernel):
    res = convolve2d(image, kernel, mode='same', boundary='symm')
    return np.clip(np.abs(res), 0, 1)


def create_bayer_mosaic(image):
    H, W, _ = image.shape
    H = H - (H % 2)
    W = W - (W % 2)
    image = image[:H, :W, :]

    mosaic = np.zeros((H, W))
    mosaic[0::2, 0::2] = image[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = image[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = image[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = image[1::2, 1::2, 2]

    return mosaic


def demosaic_convolution(mosaic):
    H, W = mosaic.shape

    R_mask = np.zeros_like(mosaic)
    G_mask = np.zeros_like(mosaic)
    B_mask = np.zeros_like(mosaic)

    R_mask[0::2, 0::2] = 1
    G_mask[0::2, 1::2] = 1
    G_mask[1::2, 0::2] = 1
    B_mask[1::2, 1::2] = 1

    R_raw = mosaic * R_mask
    G_raw = mosaic * G_mask
    B_raw = mosaic * B_mask

    K_RB = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 4.0

    K_G = np.array([
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]
    ]) / 4.0

    R_rec = convolve2d(R_raw, K_RB, mode='same', boundary='symm')
    G_rec = convolve2d(G_raw, K_G, mode='same', boundary='symm')
    B_rec = convolve2d(B_raw, K_RB, mode='same', boundary='symm')

    return np.clip(np.dstack((R_rec, G_rec, B_rec)), 0, 1)


def run_filters_demo(filename):
    img = io.imread(filename)

    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    img = img.astype(float)
    if img.max() > 1.0:
        img /= 255.0

    filters = [
        (Sx, "Sobel X"),
        (Sy, "Sobel Y"),
        (L, "Laplace"),
        (Px, "Prewitt X"),
        (G, "Gaussian"),
        (W, "Sharpen")
    ]

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    results = {}

    for i, (kernel, name) in enumerate(filters):
        res = apply_convolution(img, kernel)
        results[name] = res

        plt.subplot(3, 3, i + 2)
        plt.imshow(res, cmap='gray')
        plt.title(name)
        plt.axis('off')

    sobel_comb = np.sqrt(results["Sobel X"] ** 2 + results["Sobel Y"] ** 2)
    sobel_comb = np.clip(sobel_comb, 0, 1)

    plt.subplot(3, 3, 8)
    plt.imshow(sobel_comb, cmap='gray')
    plt.title("Sobel Magnitude")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("wyniki_filtry.png")
    plt.show()


def run_demosaicing_demo(filename):
    img_rgb = io.imread(filename)

    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 4:
        img_rgb = color.rgba2rgb(img_rgb)
    elif len(img_rgb.shape) == 2:
        print("Error: Image for demosaicing must be color (RGB).")
        return

    img_rgb = img_rgb.astype(float)
    if img_rgb.max() > 1.0:
        img_rgb /= 255.0

    mosaic = create_bayer_mosaic(img_rgb)
    img_rec = demosaic_convolution(mosaic)

    H, W, _ = img_rec.shape
    img_rgb = img_rgb[:H, :W, :]

    diff = np.abs(img_rgb - img_rec)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mosaic, cmap='gray')
    plt.title("Bayer Mosaic")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(img_rec)
    plt.title("Reconstructed")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(np.clip(diff * 5, 0, 1))
    plt.title("Difference (x5)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("wyniki_demozaikowanie.png")
    plt.show()


if __name__ == "__main__":
    file_bw = '1.jpg'
    file_col = '2.jpg'

    run_filters_demo(file_bw)
    run_demosaicing_demo(file_col)