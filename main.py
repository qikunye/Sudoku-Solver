import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Reshape, Activation
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('sudoku.csv')



# Convert puzzle strings to 9x9 arrays of integers
puzzles = [list(map(int, list(p))) for p in data['quizzes']]
solutions = [list(map(int, list(s))) for s in data['solutions']]

# Convert to numpy arrays for TensorFlow
X = np.array(puzzles).reshape(-1, 9, 9, 1)        # shape: (num_puzzles, 9, 9, 1)
y = np.array(solutions).reshape(-1, 81)           # shape: (num_puzzles, 81)

# Normalize inputs to [-0.5, 0.5]
X = X / 9.0 - 0.5

# Convert outputs to 0-8 class labels per cell
y = y - 1

model = Sequential([
    Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9, 9, 1)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'),
    Flatten(),
    Dense(81 * 9),            # 81 cells * 9 classes each
    Reshape((81, 9)),         # reshape output into 81x9
    Activation('softmax')     # softmax for probabilities over 9 classes per cell
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(
    X, y, 
    epochs=5, 
    batch_size=32, 
    validation_split=0.1,
    verbose=1
)

def solve_sudoku(model, puzzle_str):
    """
    Solve a Sudoku puzzle using a trained model.
    :param model: Trained Keras model that outputs (81,9) predictions.
    :param puzzle_str: String of length 81 with digits and '0' for blanks.
    :return: Solved puzzle as a 9x9 numpy array of digits.
    """
    # Prepare the board as a numpy array
    puzzle = [int(c) for c in puzzle_str.strip()]  # convert string to list of ints
    board = np.array(puzzle).reshape(9, 9)
    
    # Iteratively fill in blanks
    while True:
        # If no blanks (0) left, puzzle is solved
        if 0 not in board:
            break
        
        # Predict with the model on the current board
        model_input = (board / 9.0 - 0.5).reshape(1, 9, 9, 1)  # normalize like training
        predictions = model.predict(model_input)[0]  # shape (81, 9) for one puzzle
        predictions = predictions.reshape(81, 9)
        
        # Get the model's predicted digit (1-9) and confidence for each cell
        pred_digits = np.argmax(predictions, axis=1) + 1   # +1 to convert 0-8 back to 1-9
        pred_confidence = np.max(predictions, axis=1)      # highest probability for each cell
        
        # Among the blank cells, find the one with highest confidence prediction
        flat_board = board.flatten()
        blank_idxs = np.where(flat_board == 0)[0]          # indices of blanks in flattened board
        if len(blank_idxs) == 0:
            break  # no blanks
        # Choose the blank index with max confidence
        best_idx = blank_idxs[np.argmax(pred_confidence[blank_idxs])]
        best_digit = pred_digits[best_idx]
        
        # Fill the chosen cell with the predicted digit
        board.flat[best_idx] = best_digit
        # Loop continues to fill next blank
    return board


# Example unsolved puzzle (0 represents blanks)
example_puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
solved_board = solve_sudoku(model, example_puzzle)
print(solved_board)
