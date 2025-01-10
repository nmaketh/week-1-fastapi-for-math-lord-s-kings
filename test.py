from fastapi import FastAPI
import uvicorn
import numpy as np

app = FastAPI()

# Initialize M and B as specific np arrays
M = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25]])

B = np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]])

def f(x):
    """
    Compute the result of Y = M * X + B using numpy.
    Args:
        x (np.ndarray): Input matrix
    Returns:
        np.ndarray: Resulting matrix Y
    """
    return np.dot(M, x) + B

def f_without_numpy(x):
    """
    Compute the result of Y = M * X + B without numpy.
    Args:
        x (list[list[float]]): Input matrix as nested lists
    Returns:
        list[list[float]]: Resulting matrix Y
    """
    y = []
    for i in range(len(M)):
        row = []
        for j in range(len(x[0])):
            value = sum(M[i][k] * x[k][j] for k in range(len(x))) + B[i][j]
            row.append(value)
        y.append(row)
    return y

# Initialize x as a 5x5 matrix
x = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25]])

# Make a call to the function
result_with_numpy = f(x)

# Recreate the function with the sigmoid function
def sigmoid(x):
    """
    Apply the sigmoid function element-wise to a matrix.
    Args:
        x (np.ndarray): Input matrix
    Returns:
        np.ndarray: Matrix after applying the sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def f_with_sigmoid(x):
    """
    Compute Y = sigmoid(M * X + B).
    Args:
        x (np.ndarray): Input matrix
    Returns:
        np.ndarray: Resulting matrix Y
    """
    return sigmoid(np.dot(M, x) + B)

result_with_sigmoid = f_with_sigmoid(x)

@app.get("/")
def read_root():
    """Default route for testing the API."""
    return {
        "message": "FastAPI matrix operations API is running!",
        "result_with_numpy": result_with_numpy.tolist(),
        "result_with_sigmoid": result_with_sigmoid.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app)
