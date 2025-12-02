import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse


def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(x - 1)

def f3(x):
    return np.sign(np.sin(8 * x))

def h1(t):
    return np.where((t >= 0) & (t < 1), 1.0, 0.0)

def h2(t):
    return np.where((t >= -0.5) & (t < 0.5), 1.0, 0.0)

def h3(t):
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0.0)

def h4(t):
    result = np.zeros_like(t, dtype=float)
    mask = t != 0
    result[mask] = np.sin(np.pi * t[mask]) / (np.pi * t[mask])
    result[~mask] = 1.0
    return result

def interpolate_with_kernel(x_sparse, y_sparse, x_dense, kernel_func):
    y_dense = np.zeros_like(x_dense)
    for i, x_val in enumerate(x_dense):
        distances = np.abs(x_val - x_sparse)
        kernel_vals = kernel_func(distances)
        kernel_sum = np.sum(kernel_vals)
        if kernel_sum > 1e-10:
            y_dense[i] = np.sum(y_sparse * kernel_vals) / kernel_sum
        else:
            nearest_idx = np.argmin(distances)
            y_dense[i] = y_sparse[nearest_idx]
    return y_dense

def compute_mse_table():
    functions = [(f1, "f_1"), (f2, "f_2"), (f3, "f_3")]
    kernels = [(h1, "h_1"), (h2, "h_2"), (h3, "h_3"), (h4, "h_4")]
    multipliers = [2, 4, 10, 16]

    x_sparse = np.linspace(-3, 3, 100)

    print("\n" + "=" * 110)
    print("TABELA 1: MSE dla interpolacji funkcji")
    print("=" * 110)
    print(f"{'Funkcja':<15} {'Jądro':<8} {'×2':<16} {'×4':<16} {'×10':<16} {'×16':<16}")
    print("-" * 110)

    for func, func_name in functions:
        y_sparse = func(x_sparse)
        for kernel, kernel_name in kernels:
            row = [func_name, kernel_name]
            for mult in multipliers:
                x_dense = np.linspace(-3, 3, 100 * mult)
                y_interp = interpolate_with_kernel(x_sparse, y_sparse, x_dense, kernel)
                y_true = func(x_dense)
                error = mse(y_true, y_interp)
                row.append(f"{error:.6f}")
            print(f"{func_name:<15} {kernel_name:<8} {row[2]:<16} {row[3]:<16} {row[4]:<16} {row[5]:<16}")

    print("=" * 110)

def compare_methods():
    print("\n" + "=" * 70)
    print("Porównanie: jednokrokownie (×16) vs kaskada (4×2)")
    print("=" * 70)

    x_sparse = np.linspace(-3, 3, 100)
    y_sparse = f1(x_sparse)
    x_dense = np.linspace(-3, 3, 1600)
    y_true = f1(x_dense)

    y_direct = interpolate_with_kernel(x_sparse, y_sparse, x_dense, h3)
    error_direct = mse(y_true, y_direct)

    current_x = x_sparse
    current_y = y_sparse
    for _ in range(4):
        new_x = np.linspace(-3, 3, len(current_x) * 2)
        current_y = interpolate_with_kernel(current_x, current_y, new_x, h3)
        current_x = new_x
    y_cascade = current_y[:len(x_dense)]
    error_cascade = mse(y_true, y_cascade)

    print(f"{'Metoda':<30} {'MSE':<15}")
    print("-" * 70)
    print(f"{'Jednokrokowna (×16)':<30} {error_direct:.6f}")
    print(f"{'Kaskada (4×2)':<30} {error_cascade:.6f}")
    print("=" * 70)

def plot_interpolation():
    x_sparse = np.linspace(-3, 3, 15)
    x_dense = np.linspace(-3, 3, 500)

    functions = [(f1, "f_1(x) = sin(x)"), (f2, "f_2(x) = sin(x-1)"), (f3, "f_3(x) = sgn(sin(8x))")]
    kernels = [(h1, "h_1"), (h3, "h_3"), (h4, "h_4")]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for func, func_title in functions:
        y_sparse = func(x_sparse)
        y_true = func(x_dense)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Interpolacja funkcji {func_title}", fontsize=14, fontweight='bold')

        for idx, (kernel, kernel_name) in enumerate(kernels):
            y_interp = interpolate_with_kernel(x_sparse, y_sparse, x_dense, kernel)

            axes[idx].plot(x_dense, y_true, 'b-', linewidth=2.5, label='Oryginalna', alpha=0.8)
            axes[idx].plot(x_dense, y_interp, color=colors[idx], linewidth=2, label='Interpolacja', alpha=0.8)
            axes[idx].scatter(x_sparse, y_sparse, color=colors[idx], s=60, zorder=5, edgecolors='black',
                              linewidth=1)
            axes[idx].set_title(f"Jądro {kernel_name}")
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(-1.3, 1.3)

        plt.tight_layout()
        func_name = func_title.split('(')[0].strip().lower()
        plt.savefig(f'interpolation_{func_name}.png', dpi=100, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    compute_mse_table()
    compare_methods()
    plot_interpolation()
