import string

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def null_step_5(matrix: np.ndarray):
    matrix.flat[::5] = 0
    return matrix

def null_step_diagonal(matrix: np.ndarray):
    np.fill_diagonal(matrix, 0)
    return matrix


# Function to visualize a matrix using a heatmap.
def visualize_matrix(matrix: np.ndarray , title: string):
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()


# Function to find the median value of a matrix.
def find_median(matrix: np.ndarray) -> ndarray:
    return np.median(matrix)


# Function to find the average (mean) value of a matrix.
def find_average(matrix: ndarray) -> ndarray:
    return np.mean(matrix)


def get_gold_value(mean_value: float, average_value: float):
    return (mean_value + average_value) / 2


def fill_zeros_with_mean(matrix: np.ndarray, gold_value: float) -> np.ndarray:
    matrix[matrix == 0] = gold_value
    return matrix


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
    rows = 13
    cols = 17

    # Generate a random matrix.
    matrix = generate_matrix(rows, cols)
    print(matrix)
    visualize_matrix(matrix, "original matrix")

    # Ask the user to provide a value.
    value = float(input("Please, enter value to check: "))
    print("The nearest value in matrix is:", find_nearest_value(matrix, value))

    # Find and display the average value of the matrix.
    average = find_average(matrix)
    print(f"The average value of the matrix is: {average}")

    # Find and display the median value of the matrix.
    median = find_median(matrix)
    print(f"The median value of the matrix is: {median}")

    gold_value = get_gold_value(median, average)


    # Normalize the matrix so that values range between 0 and 1.
    min_value = matrix.min()
    max_value = matrix.max()
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    print("\nNormalized matrix:")
    print(normalized_matrix)
    visualize_matrix(matrix, "normalized")


# Execute the main function when the script is run.
if __name__ == '__main__':
    main()
