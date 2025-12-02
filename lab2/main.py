from image import image_scaling
from inter import compute_mse_table, compare_methods, plot_interpolation

if __name__ == "__main__":
    compute_mse_table()
    compare_methods()
    plot_interpolation()
    image_scaling("oar2.jpg")