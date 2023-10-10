import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel

# Load an image from the given filepath and ensure its size is 600x600 or larger.
def load_image(filepath):
    img = plt.imread(filepath)
    if img.shape[0] < 600 or img.shape[1] < 600:
        raise ValueError("Зображення повинно мати розмір 600x600 або більше")
    return img

# Display the image with an optional title and colormap.
def show_image(image, title="", cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Hide axis labels
    plt.show()

# Enhance the contrast of the image.
def enhance_contrast(img):
    # Find the minimum and maximum values for each color channel.
    min_val = img.min(axis=(0, 1), keepdims=True)
    max_val = img.max(axis=(0, 1), keepdims=True)
    # Normalize and scale the image to enhance contrast.
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Convert the color image to grayscale.
def convert_to_grayscale(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

# Split the image into its red, green, and blue channels and display them.
def split_into_channels(img):
    channels = ["Reds", "Greens", "Blues"]
    plt.figure(figsize=(12, 6))

    for idx, channel in enumerate(channels):
        plt.subplot(1, 3, idx+1)
        plt.imshow(img[:, :, idx], cmap=channel, vmin=0, vmax=255)
        plt.title(f"{channel[:-1]} Channel")
        plt.axis('off')

    plt.show()

# Apply edge detection to a grayscale image.
def edge_detection(img_gray):
    # Compute the Sobel gradient for the x and y directions.
    sx = sobel(img_gray, axis=0, mode='constant')
    sy = sobel(img_gray, axis=1, mode='constant')
    # Combine the two gradients to get the edge intensity.
    return np.hypot(sx, sy)

# Apply a Gaussian blur to the image.
def blur_image(img, sigma=10):
    return gaussian_filter(img, sigma=(sigma, sigma, 0))

def main():
    image_path = 'audi2.jpg'
    image = load_image(image_path)

    # Display various versions of the image using the defined functions.
    show_image(image, "Оригінальне зображення")
    show_image(enhance_contrast(image), "З контрастністю")
    split_into_channels(image)
    show_image(convert_to_grayscale(image), "Чорно-біле зображення", cmap="gray")
    show_image(blur_image(image), "Розмите зображення")
    show_image(edge_detection(convert_to_grayscale(image)), "Edge Detection", cmap="gray")

# Execute the main function when the script is run.
if __name__ == "__main__":
    main()
