# 🧩 Sudoku Solver with TensorFlow

A beginner-friendly **Sudoku solver powered by a Convolutional Neural Network (CNN)** built with [TensorFlow/Keras](https://www.tensorflow.org/).

This project trains a model to **fill in Sudoku puzzles** from a CSV dataset of puzzles and solutions. Instead of hand-coding Sudoku rules, the solver *learns* the patterns directly from millions of examples. The result: an AI solver that completes puzzles step by step, much like a human.

---

## 🚀 Features

* **Convolutional Neural Network (CNN):** Learns Sudoku rules by scanning the board like an image.
* **Iterative Solver:** Fills in the puzzle one cell at a time, always choosing the model’s most confident guess.
* **Beginner-Friendly Design:** Code and comments written so users with minimal ML experience can follow along.
* **Dataset Ready:** Works with the [Kaggle Sudoku dataset](https://www.kaggle.com/bryanpark/sudoku).

---

## 📂 Project Structure

```
sudoku-solver/
│── sudoku_solver.py      # Script version of the solver
│── sudoku.csv            # Dataset (downloaded from Kaggle)
│── README.md             # Project documentation
```

---

## 📊 How It Works

### 1. Data Preparation

* Puzzles and solutions are loaded from `sudoku.csv` (81-character strings).
* Each puzzle is reshaped into a **9x9 grid** where `0` means blank.
* Solutions are converted into class labels (digits 1–9).

### 2. Neural Network Model

* **Input:** 9×9 puzzle grid (treated like a tiny grayscale image).
* **Conv Layers:** Learn local Sudoku rules (e.g., no duplicate numbers in a row/col/box).
* **Dense Layer:** Produces predictions for each cell.
* **Output:** 81 sets of 9 probabilities → one probability distribution for each cell.

### 3. Training

* Model learns by comparing predictions with the true solution.
* Uses `sparse_categorical_crossentropy` loss.
* Learns Sudoku rules after seeing many puzzles.

### 4. Iterative Solver

Instead of trusting the model’s first guess for all cells, we:

1. Predict probabilities for every blank cell.
2. Fill the cell with the **highest confidence prediction**.
3. Update the puzzle and repeat until solved.

This mimics human solving: *“Do the easy cells first, then move on.”*

---

## 💻 Installation & Setup

1. Clone this repo:

   ```bash
   git clone https://github.com/your-username/sudoku-solver.git
   cd sudoku-solver
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   (or install manually: `tensorflow pandas numpy scikit-learn matplotlib`)

3. Download the dataset from Kaggle:
   [Sudoku Dataset (1M puzzles)](https://www.kaggle.com/bryanpark/sudoku)
   Place `sudoku.csv` in the project root.

---

## ▶️ Usage

### Train the model:

```bash
python sudoku_solver.py --train
```

### Solve a puzzle:

```bash
python sudoku_solver.py --solve "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Example output:

```
[[5 3 4 6 7 8 9 1 2]
 [6 7 2 1 9 5 3 4 8]
 [1 9 8 3 4 2 5 6 7]
 ...
]
```

---

## 📸 Example

Input puzzle (zeros = blanks):

```
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
```

Output solution:

```
534678912
672195348
198342567
859761423
426853791
713924856
961537284
287419635
345286179
```

---

## 📚 References

* Dataset: [Kaggle – Sudoku Puzzles](https://www.kaggle.com/bryanpark/sudoku)
* TensorFlow Documentation: [Convolutional Layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
* Community Projects:

  * [Kaggle Notebook: CNN Sudoku Solver](https://www.kaggle.com/code/subhraneelpaul/sudoku-solver-using-cnn)
  * [GeeksforGeeks: Solving Sudoku using Neural Networks](https://www.geeksforgeeks.org/solving-sudoku-using-neural-networks/)

---

## 🧠 Learning Takeaways

This project is designed for learners:

* Understand how CNNs can be applied beyond images.
* Practice data preprocessing, model training, and inference in TensorFlow.
* Learn an **iterative ML-driven problem-solving strategy**.

---

## 🔮 Future Work

* Extend to **image-based Sudoku puzzles** (using OCR + digit recognition).
* Experiment with different architectures (e.g., recurrent models, transformers).
* Deploy as a simple web app where users upload a puzzle to get a solution.

---

## 🙌 Acknowledgements

Thanks to the open-source community and Kaggle contributors for datasets and inspiration.


