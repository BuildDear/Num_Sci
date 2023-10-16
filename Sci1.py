# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image, ImageEnhance


def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from a given filepath and ensure its size is 600x600 or larger.
    """
    img = plt.imread(filepath)
    if img.shape[0] < 600 or img.shape[1] < 600:
        raise ValueError("Зображення повинно мати розмір 600x600 або більше")
    return img


def show_image(image: np.ndarray, title: str = "", cmap: str = None):
    """
    Display an image with an optional title and colormap.
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def adjust_contrast(img: Image.Image, level: int) -> Image.Image:
    """
    Adjust the contrast of an image based on the provided level.
    """
    if not 1 <= level <= 10:
        raise ValueError("Рівень контрасту повинен бути від 1 до 10")

    enhancer = ImageEnhance.Contrast(img)
    factor = 1 + (level - 5) * 0.2
    return enhancer.enhance(factor)


def numpy_to_pil(image_np: np.ndarray) -> Image.Image:
    """
    Convert a numpy image array to a PIL Image object.
    """
    return Image.fromarray((image_np * 255).astype(np.uint8))


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a color image to grayscale.
    """
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def split_into_channels(img: np.ndarray):
    """
    Split the image into its RGB channels and display them.
    """
    channels = ["Reds", "Greens", "Blues"]
    plt.figure(figsize=(12, 6))

    for idx, channel in enumerate(channels):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(img[:, :, idx], cmap=channel, vmin=0, vmax=255)
        plt.title(f"{channel[:-1]} Channel")
        plt.axis('off')

    plt.show()


def edge_detection(img_gray: np.ndarray) -> np.ndarray:
    """
    Apply edge detection to a grayscale image.
    """
    # Compute the Sobel gradient for the x and y directions.
    sx = sobel(img_gray, axis=0, mode='constant')
    sy = sobel(img_gray, axis=1, mode='constant')

    # Combine the two gradients to get the edge intensity.
    return np.hypot(sx, sy)


def blur_image(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    """
    return gaussian_filter(img, sigma=(sigma, sigma, 0))


def main():
    """
    Main function to load, process and display various versions of an image.
    """
    image_path = 'audi2.jpg'
    image = load_image(image_path)

    # Display the original image
    show_image(image, "Оригінальне зображення")

    # Adjust contrast based on user input and display the image
    level = int(input("Enter level of contrast from 1 to 10. 5 - middle"))
    image_pil = numpy_to_pil(image)
    show_image(adjust_contrast(image_pil, level), "З контрастністю")

    # Display individual RGB channels
    split_into_channels(image)

    # Convert to grayscale and display
    show_image(convert_to_grayscale(image), "Чорно-біле зображення", cmap="gray")

    # Blur and display the image
    show_image(blur_image(image), "Розмите зображення")

    # Apply edge detection and display the result
    show_image(edge_detection(convert_to_grayscale(image)), "Edge Detection", cmap="gray")


# Execute the main function when the script is run.
if __name__ == "__main__":
    main()
