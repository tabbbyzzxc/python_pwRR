import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from sklearn.metrics import mean_squared_error as mse
from skimage import io, color, data

def f1(x):
    return np.sin(x)


def f2(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.sin(1.0 / x)
        val[x == 0] = 0.0
    return val


def f3(x):
    return np.sign(np.sin(8 * x))


def h1(t):
    return np.where((t >= -0.5) & (t < 0.5), 1.0, 0.0)


def h3(t):
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0.0)


def h4(t):
    result = np.zeros_like(t, dtype=float)
    mask = np.abs(t) < 1e-9
    result[mask] = 1.0
    not_mask = ~mask
    t_nm = t[not_mask]
    result[not_mask] = np.sin(np.pi * t_nm) / (np.pi * t_nm)
    return result


def interpolate_with_kernel(x_sparse, y_sparse, x_dense, kernel_func):
    y_dense = np.zeros_like(x_dense)

    if len(x_sparse) > 1:
        step = x_sparse[1] - x_sparse[0]
    else:
        step = 1.0

    for i, x_val in enumerate(x_dense):
        dist = x_val - x_sparse
        t = dist / step
        weights = kernel_func(t)
        y_dense[i] = np.sum(y_sparse * weights)

    return y_dense


def manual_downscale_avg(img, factor=2):
    k_size = int(factor)
    kernel = np.ones((k_size, k_size)) / (k_size ** 2)

    blurred = convolve2d(img, kernel, mode='same', boundary='symm')

    return blurred[::k_size, ::k_size]


def manual_upscale_2d(img, factor, kernel_func):
    H, W = img.shape
    new_H, new_W = int(H * factor), int(W * factor)

    temp_img = np.zeros((H, new_W))
    x_sparse = np.arange(W)
    x_dense = np.linspace(0, W - 1, new_W)

    for r in range(H):
        temp_img[r, :] = interpolate_with_kernel(x_sparse, img[r, :], x_dense, kernel_func)

    final_img = np.zeros((new_H, new_W))
    y_sparse = np.arange(H)
    y_dense = np.linspace(0, H - 1, new_H)

    for c in range(new_W):
        final_img[:, c] = interpolate_with_kernel(y_sparse, temp_img[:, c], y_dense, kernel_func)

    return final_img


def run_part1_functions():
    x_sparse = np.linspace(-3, 3, 40)
    x_dense = np.linspace(-3, 3, 500)

    funcs = [
        (f1, "f1(x)=sin(x)"),
        (f2, "f2(x)=sin(1/x)"),
        (f3, "f3(x)=sgn(sin(8x))")
    ]
    kernels = [(h1, "h1 (Box)"), (h3, "h3 (Liniowe)"), (h4, "h4 (Sinc)")]

    print(f"{'Funkcja':<10} {'Jądro':<15} {'MSE':<15}")
    print("-" * 40)

    for fname, ftitle in funcs:
        y_sparse = fname(x_sparse)
        y_true = fname(x_dense)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Interpolacja: {ftitle}", fontsize=14)

        for i, (kfunc, kname) in enumerate(kernels):
            y_interp = interpolate_with_kernel(x_sparse, y_sparse, x_dense, kfunc)
            err = mse(y_true, y_interp)
            print(f"{ftitle[:5]:<10} {kname:<15} {err:.6f}")

            axes[i].plot(x_dense, y_true, 'b-', alpha=0.3, label='Oryginał')
            axes[i].plot(x_dense, y_interp, 'r-', label='Interpolacja')
            axes[i].scatter(x_sparse, y_sparse, c='k', s=10, label='Próbki')
            axes[i].set_title(f"{kname}\nMSE={err:.4f}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-1.5, 1.5)

        plt.tight_layout()
        plt.savefig(f"interpolacja_{ftitle[:2]}.png")
        plt.show()


def run_part2_images():
    nazwa_pliku = 'oar2.jpg'

    try:
        print(f"Wczytywanie pliku: {nazwa_pliku}...")
        img = io.imread(nazwa_pliku)
    except FileNotFoundError:
        img = data.camera()
    except Exception as e:
        return

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        img = color.rgb2gray(img)

    img = img.astype(float)
    if img.max() > 1.0:
        img /= 255.0

    print(f"Obraz gotowy. Rozmiar: {img.shape}")

    img_small = manual_downscale_avg(img, factor=2)

    kernels = [(h1, "h1 (Najbliższy sąsiad)"), (h3, "h3 (Liniowe)"), (h4, "h4 (Sinc)")]

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Oryginał\n{img.shape}")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_small, cmap='gray')
    plt.title(f"Pomniejszony (Średnia)\n{img_small.shape}")
    plt.axis('off')

    print("-" * 60)
    print(f"{'Metoda (Jądro)':<30} {'MSE (Rekonstrukcja)':<20}")
    print("-" * 60)

    for i, (kfunc, kname) in enumerate(kernels):
        img_rec = manual_upscale_2d(img_small, 2, kfunc)

        H, W = img.shape
        img_rec = img_rec[:H, :W]

        err = mse(img, img_rec)
        print(f"Wynik MSE: {err:.6f}")

        plt.subplot(2, 3, i + 4)
        plt.imshow(img_rec, cmap='gray')
        plt.title(f"{kname}\nMSE={err:.5f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("skalowanie_wynik.png")
    plt.show()


if __name__ == "__main__":
    run_part1_functions()
    run_part2_images()