import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

# Function to zero out the diagonal of a matrix and return the original diagonal values.
def zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    # Copy the original diagonal values.
    diagonal_values = np.diagonal(matrix).copy()
    # Fill the diagonal of the matrix with zeros.
    np.fill_diagonal(matrix, 0)
    # Return the original diagonal values.
    return diagonal_values

# Function to restore the diagonal of a matrix using given diagonal values.
def restore_diagonal(matrix: np.ndarray, diagonal_values: np.ndarray) -> np.ndarray:
    # Fill the diagonal of the matrix with the provided diagonal values.
    np.fill_diagonal(matrix, diagonal_values)
    return matrix

# Function to visualize a matrix using a heatmap.
def visualize_matrix(matrix: np.ndarray):
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Matrix Visualization")
    plt.show()

# Function to find the median value of a matrix.
def find_median(matrix: np.ndarray) -> ndarray:
    return np.median(matrix)

# Function to find the average (mean) value of a matrix.
def find_average(matrix: ndarray) -> ndarray:
    return np.mean(matrix)

# Function to find the nearest value to a given value in the matrix.
def find_nearest_value(matrix: ndarray, value: float) -> float:
    # Calculate the absolute difference between each matrix element and the given value.
    diff = np.abs(matrix - value)
    # Find the index of the minimum difference.
    index = np.unravel_index(diff.argmin(), diff.shape)
    return matrix[index]

# Function to generate a random integer matrix with values between 0 and 100.
def generate_matrix(rows: int, cols: int) -> ndarray:
    matrix = np.random.randint(0, 101, (rows, cols))
    return matrix

def main():
    # Number of rows and columns for the matrix.
    rows = 19
    cols = 16

    # Generate a random matrix.
    matrix = generate_matrix(rows, cols)
    print(matrix)

    # Ask the user to provide a value.
    value = float(input("Please, enter value to check: "))
    # Find the nearest value in the matrix to the given value.
    nearest_value = find_nearest_value(matrix, value)
    print("The nearest value in matrix is:", nearest_value)

    # Find and display the average and median values of the matrix.
    average = find_average(matrix)
    print(f"The average value of the matrix is: {average}")
    median = find_median(matrix)
    print(f"The median value of the matrix is: {median}")

    # Visualize the matrix using a heatmap.
    visualize_matrix(matrix)

    # Zero out the diagonal of the matrix and then restore it.
    saved_diagonal = zero_diagonal(matrix)
    restore_diagonal(matrix, saved_diagonal)
    print("\nRestored matrix:")
    print(matrix)

    # Normalize the matrix so that values range between 0 and 1.
    min_value = matrix.min()
    max_value = matrix.max()
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    print("\nNormalized matrix:")
    print(normalized_matrix)

# Execute the main function when the script is run.
if __name__ == '__main__':
    main()
