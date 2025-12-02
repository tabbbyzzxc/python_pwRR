import numpy as np
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error as mse
from skimage import io, color
import matplotlib.pyplot as plt


def image_scaling(path):
    img = io.imread(path)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    img = (img * 255).astype(np.uint8)

    print("\n" + "=" * 70)
    print("TABELA 2: Skalowanie obrazów")
    print("=" * 70)
    print(f"{'Metoda':<20} {'MSE (downscale)':<20} {'MSE (upscale)':<20}")
    print("-" * 70)

    img_small_bilinear = zoom(img.astype(float), 0.5, order=1)
    img_small_cubic = zoom(img.astype(float), 0.5, order=3)
    img_small_lanczos = zoom(img.astype(float), 0.5, order=4)

    img_down_bilinear = zoom(img_small_bilinear, 2.0, order=1)[:img.shape[0], :img.shape[1]]
    img_down_cubic = zoom(img_small_cubic, 2.0, order=3)[:img.shape[0], :img.shape[1]]
    img_down_lanczos = zoom(img_small_lanczos, 2.0, order=4)[:img.shape[0], :img.shape[1]]

    mse_down_bilinear = mse(img.astype(float), img_down_bilinear)
    mse_down_cubic = mse(img.astype(float), img_down_cubic)
    mse_down_lanczos = mse(img.astype(float), img_down_lanczos)

    img_big_bilinear = zoom(img.astype(float), 2.0, order=1)
    img_big_cubic = zoom(img.astype(float), 2.0, order=3)
    img_big_lanczos = zoom(img.astype(float), 2.0, order=4)

    mse_up_bilinear = mse(img.astype(float), img_big_bilinear[:img.shape[0], :img.shape[1]])
    mse_up_cubic = mse(img.astype(float), img_big_cubic[:img.shape[0], :img.shape[1]])
    mse_up_lanczos = mse(img.astype(float), img_big_lanczos[:img.shape[0], :img.shape[1]])

    print(f"{'Biliniowa':<20} {mse_down_bilinear:<20.6f} {mse_up_bilinear:.6f}")
    print(f"{'Cubic':<20} {mse_down_cubic:<20.6f} {mse_up_cubic:.6f}")
    print(f"{'Lanczos':<20} {mse_down_lanczos:<20.6f} {mse_up_lanczos:.6f}")
    print("=" * 70)

    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    axes1[0].imshow(img, cmap='gray')
    axes1[0].set_title("Oryginał")
    axes1[0].axis('off')
    axes1[1].imshow(img_small_bilinear, cmap='gray')
    axes1[1].set_title(f"Downscale\nMSE={mse_down_bilinear:.6f}")
    axes1[1].axis('off')
    axes1[2].imshow(img_big_bilinear, cmap='gray')
    axes1[2].set_title(f"Biliniowa\nMSE={mse_up_bilinear:.6f}")
    axes1[2].axis('off')
    plt.tight_layout()
    plt.savefig('image_scaling_biliniowa.png', dpi=100, bbox_inches='tight')

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    axes2[0].imshow(img, cmap='gray')
    axes2[0].set_title("Oryginał")
    axes2[0].axis('off')
    axes2[1].imshow(img_small_cubic, cmap='gray')
    axes2[1].set_title(f"Downscale\nMSE={mse_down_cubic:.6f}")
    axes2[1].axis('off')
    axes2[2].imshow(img_big_cubic, cmap='gray')
    axes2[2].set_title(f"Cubic\nMSE={mse_up_cubic:.6f}")
    axes2[2].axis('off')
    plt.tight_layout()
    plt.savefig('image_scaling_cubic.png', dpi=100, bbox_inches='tight')

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    axes3[0].imshow(img, cmap='gray')
    axes3[0].set_title("Oryginał")
    axes3[0].axis('off')
    axes3[1].imshow(img_small_lanczos, cmap='gray')
    axes3[1].set_title(f"Downscale\nMSE={mse_down_lanczos:.6f}")
    axes3[1].axis('off')
    axes3[2].imshow(img_big_lanczos, cmap='gray')
    axes3[2].set_title(f"Lanczos\nMSE={mse_up_lanczos:.6f}")
    axes3[2].axis('off')
    plt.tight_layout()
    plt.savefig('image_scaling_lanczos.png', dpi=100, bbox_inches='tight')

    plt.show()