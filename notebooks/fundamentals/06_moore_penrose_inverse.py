import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. **System of Linear Equations**

    **Definition:**
    A system of linear equations can be written as:
    $$ A \mathbf{x} = \mathbf{b} $$
    Where:
    - $A \in \mathbb{R}^{m \times n}$ is the matrix of coefficients.
    - $\mathbf{x} \in \mathbb{R}^{n}$ is the vector of unknowns.
    - $\mathbf{b} \in \mathbb{R}^{m}$ is the right-hand side vector.

    The goal is to solve for $\mathbf{x}$. This system can represent a variety of real-world problems, including statistical models, physics simulations, or optimization problems.

    ---

    ### 2. **Rank of a Matrix**

    **Definition:**
    The **rank** of a matrix $A$, denoted as $\text{rank}(A)$, is the dimension of its column (or row) space. It represents the number of linearly independent rows or columns in the matrix.

    **Properties:**
    - If $A \in \mathbb{R}^{m \times n}$, then $\text{rank}(A) \leq \min(m, n)$.
    - A **full-rank matrix** has $\text{rank}(A) = \min(m, n)$. For square matrices, full rank implies that the matrix is invertible.

    **Importance:**
    The rank gives insight into the solvability of the linear system. A higher rank typically means the system is more constrained, leading to fewer solutions or more specific solutions.

    ---

    ### 3. **Types of Linear Systems**

    Linear systems can vary based on the number of equations and variables. These types include:

    - **Unique Solution (Full Rank, Square Matrix):**
      If $A \in \mathbb{R}^{n \times n}$ and $\text{rank}(A) = n$, the system has a unique solution:
      $$ \mathbf{x} = A^{-1} \mathbf{b} $$

    - **Overdetermined System (More Equations than Variables):**
      If $m > n$, the system may be inconsistent (no solution), the least squares solution minimizes the error $\| A \mathbf{x} - \mathbf{b} \|_2^2$:
      $$ \mathbf{x} = (A^T A)^{-1} A^T \mathbf{b} $$

    - **Underdetermined System (More Variables than Equations):**
      If $m < n$, the system typically has infinitely many solutions. The minimum norm solution is found using the Moore-Penrose pseudoinverse:
      $$ \mathbf{x} = A^+ \mathbf{b} $$

    ---

    ### 4. **Relationship Between Rank and Determinacy**

    The rank of the matrix $A$ determines the nature of the solution to the system $A \mathbf{x} = \mathbf{b}$:

    - **Full Rank (Square Matrix):**
      If $A \in \mathbb{R}^{n \times n}$ and $\text{rank}(A) = n$, the system has a unique solution (if consistent).

    - **Less Than Full Rank:**
      If $\text{rank}(A) < n$, the system may have infinitely many solutions (if consistent) or no solution at all.
      - If $\text{rank}(A) < m$, the system may be inconsistent.

    The rank is a key indicator of the solvability of the system, guiding us toward the appropriate method for finding solutions.

    ---

    ### 5. **Moore-Penrose Inverse**

    The **Moore-Penrose pseudoinverse** $A^+$ is a generalized inverse of a matrix $A$, and it exists for any matrix $A \in \mathbb{R}^{m \times n}$, whether square or rectangular. It satisfies four properties:
    1. $A A^+ A = A$
    2. $A^+ A A^+ = A^+$
    3. $(A A^+)^T = A A^+$
    4. $(A^+ A)^T = A^+ A$

    For matrices with full rank:
    - If $A$ has full column rank:
      $$ A^+ = (A^T A)^{-1} A^T $$
    - If $A$ has full row rank:
      $$ A^+ = A^T (A A^T)^{-1} $$

    The Moore-Penrose inverse helps solve linear systems, even when they do not have a unique solution (as in overdetermined or underdetermined systems).

    ---

    ### 6. **How Moore-Penrose Inverse Solves Systems**

    The Moore-Penrose pseudoinverse $A^+$ can be used to solve all types of linear systems, including underdetermined and overdetermined systems.

    - **For a Unique Solution (Square, Full Rank):**
      If $A \in \mathbb{R}^{n \times n}$ and $\text{rank}(A) = n$, the Moore-Penrose inverse gives the unique solution:
      $$ \mathbf{x} = A^{-1} \mathbf{b} = A^+ \mathbf{b} $$

    - **For an Overdetermined System ($m > n$):**
      If $\text{rank}(A) = n$, the least squares solution is:
      $$ \mathbf{x} = A^+ \mathbf{b} = (A^T A)^{-1} A^T \mathbf{b} $$
      This minimizes $\| A \mathbf{x} - \mathbf{b} \|_2^2$, the sum of squared residuals.

    - **For an Underdetermined System ($m < n$):**
      If $\text{rank}(A) = m$, the minimum-norm solution is:
      $$ \mathbf{x} = A^+ \mathbf{b} $$
      This minimizes $\| \mathbf{x} \|_2$, ensuring the solution has the smallest possible magnitude.

    The Moore-Penrose inverse provides the best approximation to a solution, even in cases where the system does not have a unique solution.
    """)
    return


@app.cell
def _():
    import numpy as np
    np.random.seed(47)

    A1 = np.random.randn(3, 3)  
    b1 = np.random.randn(3)  # 3x1 vector with random values
    x1 = np.linalg.inv(A1) @ b1  # Unique solution

    # 3x2 matrix for over-determined system
    A2 = np.random.randn(3, 2)  
    b2 = np.random.randn(3)
    x2 = np.linalg.pinv(A2) @ b2  # Least squares solution

    # 2x3 matrix for under-determined system
    A3 = np.random.randn(2, 3)  
    b3 = np.random.randn(2)
    x3 = np.linalg.pinv(A3) @ b3  # Min norm solution

    def print_matrix_solution(A, b, x, label):
        print(f"\n{label} - Matrix:\n{np.array_str(A, precision=3)}")
        print(f"\n{label} - Vector:\n{np.array_str(b, precision=3)}")
        print(f"\n{label} - Solution:\n{np.array_str(x, precision=3)}")

    print_matrix_solution(A1, b1, x1, "A1 (3x3)")
    print_matrix_solution(A2, b2, x2, "A2 (3x2 - Over-determined)")
    print_matrix_solution(A3, b3, x3, "A3 (2x3 - Under-determined)")
    return


if __name__ == "__main__":
    app.run()
