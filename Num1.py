import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def null_step_5(matrix: np.ndarray) -> np.ndarray:
    """
    Set every fifth element of the matrix to zero.
    """
    matrix.flat[::5] = 0
    return matrix


def null_step_diagonal(matrix: np.ndarray) -> np.ndarray:
    """
    Set the main diagonal of the matrix to zero.
    """
    np.fill_diagonal(matrix, 0)
    return matrix


def fill_gold(matrix: np.ndarray, gold_value: float) -> np.ndarray:
    """
    Replace all zeros in the matrix with a specified gold_value.
    """
    matrix[matrix == 0] = gold_value
    return matrix


def plot_histogram(matrix, title: str):
    """
    Display a histogram of the matrix values.
    """
    flattened_data = matrix.flatten()
    plt.hist(flattened_data, bins=50, color='green', edgecolor='black')
    plt.title(title)
    plt.xlabel("Значення")
    plt.ylabel("Кількість")
    plt.show()


def plot_matrix(matrix: np.ndarray):
    """
    Display the matrix as an image.
    """
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.show()


def find_median(matrix: np.ndarray) -> ndarray:
    """
    Calculate the median value of the matrix.
    """
    return np.median(matrix)


def find_average(matrix: ndarray) -> ndarray:
    """
    Calculate the average (mean) value of the matrix.
    """
    return np.mean(matrix)


def get_gold_value(mean_value: float, average_value: float) -> float:
    """
    Calculate the average of the mean and average values.
    """
    return (mean_value + average_value) / 2


def find_nearest_value(matrix: ndarray, value: float) -> float:
    """
    Find the value in the matrix that is closest to a given value.
    """
    diff = np.abs(matrix - value)
    index = np.unravel_index(diff.argmin(), diff.shape)
    return matrix[index]


def generate_matrix(rows: int, cols: int) -> ndarray:
    """
    Generate a random matrix with integer values between 0 and 100.
    """
    return np.random.randint(0, 101, (rows, cols))


def main():
    """
    Main function to execute the matrix operations and display results.
    """
    # Generate and display a random matrix.
    rows, cols = 10, 10
    matrix = generate_matrix(rows, cols)
    print(matrix)
    plot_histogram(matrix, "Original Matrix Histogram")

    # Get user input and display the nearest value in the matrix.
    value = float(input("Please, enter value to check: "))
    print("The nearest value in matrix is:", find_nearest_value(matrix, value))

    # Display the average and median values of the matrix.
    average = find_average(matrix)
    print(f"The average value of the matrix is: {average}")
    median = find_median(matrix)
    print(f"The median value of the matrix is: {median}")

    # Compute the gold_value.
    gold_value = get_gold_value(median, average)

    # Null every fifth element, fill with gold value, and display.
    matrix_step_5 = null_step_5(matrix.copy())
    plot_histogram(matrix_step_5, "Matrix with every 5th element nullified")
    matrix_step_5 = fill_gold(matrix_step_5, gold_value)
    plot_histogram(matrix_step_5, "Matrix after filling with gold value")

    # Nullify the diagonal, fill with gold value, and display.
    matrix_step_diagonal = null_step_diagonal(matrix.copy())
    plot_histogram(matrix_step_diagonal, "Matrix with diagonal nullified")
    matrix_step_diagonal = fill_gold(matrix_step_diagonal, gold_value)
    plot_histogram(matrix_step_diagonal, "Matrix diagonal filled with gold value")

    # Normalize matrix and display.
    min_value, max_value = matrix.min(), matrix.max()
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    print("\nNormalized matrix:")
    plot_histogram(normalized_matrix, "Normalized Matrix Histogram")


if __name__ == '__main__':
    main()
