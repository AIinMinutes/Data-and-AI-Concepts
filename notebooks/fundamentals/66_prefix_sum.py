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
    # Algebra Behind Prefix Sum: 1D and 2D

    ## 1D Prefix Sum
    1D prefix sum is a technique used to compute cumulative sums efficiently. Given an array `A` of size `n`, the prefix sum array `P` is defined as:
    $$
    P[i] = A[0] + A[1] + \dots + A[i]
    $$
    for $ 0 \leq i < n $.

    ### Recursive Relation
    $$
    P[i] = P[i-1] + A[i], \quad \text{for } i > 0
    $$
    with the base case:
    $$
    P[0] = A[0]
    $$

    ### Usage
    To compute the sum of any subarray $ A[l:r] $ (inclusive):
    $$
    \text{Sum}(A[l:r]) = P[r] - P[l-1], \quad \text{for } l > 0
    $$
    If $ l = 0 $:
    $$
    \text{Sum}(A[0:r]) = P[r]
    $$

    ---

    ## 2D Prefix Sum
    2D prefix sum extends this concept to matrices. Given a matrix `M` of size $ n \times m $, the 2D prefix sum matrix `P` is defined as:
    $$
    P[i][j] = \sum_{0 \leq x \leq i, 0 \leq y \leq j} M[x][y]
    $$
    for $ 0 \leq i < n $, $ 0 \leq j < m $.

    ### Recursive Relation
    $$
    P[i][j] = M[i][j] + P[i-1][j] + P[i][j-1] - P[i-1][j-1], \quad \text{for } i, j > 0
    $$
    with base cases:
    $$
    P[i][0] = \sum_{x=0}^i M[x][0], \quad P[0][j] = \sum_{y=0}^j M[0][y], \quad P[0][0] = M[0][0]
    $$

    ### Usage
    To compute the sum of any submatrix $ M[x1:x2][y1:y2] $:
    $$
    \text{Sum}(M[x1:x2][y1:y2]) = P[x2][y2] - P[x1-1][y2] - P[x2][y1-1] + P[x1-1][y1-1]
    $$
    Where adjustments are made based on boundary conditions:
    - If $ x1 = 0 $, omit $ P[x1-1][y2] $.
    - If $ y1 = 0 $, omit $ P[x2][y1-1] $.
    - If both $ x1 = 0 $ and $ y1 = 0 $, sum equals $ P[x2][y2] $.

    ---

    ## Key Insights
    1. **Time Complexity**:
       - Constructing the prefix sum array/matrix: $ O(n) $ for 1D, $ O(n \times m) $ for 2D.
       - Querying the sum: $ O(1) $.

    2. **Avoiding Double Counting**:
       - In the 2D prefix sum formula, subtracting $ P[i-1][j] $ and $ P[i][j-1] $ removes extra sums, and adding back $ P[i-1][j-1] $ corrects for over-subtraction.
    """)
    return


@app.cell
def _():
    from typing import List, Tuple


    class PrefixSum1D:
        def __init__(self, nums: List[int]):
            """
            Initializes the prefix sum for a 1D array.

            Args:
                nums (List[int]): The input array of integers.
            """
            self.prefix = self._compute_prefix_sum(nums)

        @staticmethod
        def _compute_prefix_sum(nums: List[int]) -> List[int]:
            """
            Computes the prefix sum for the input array.

            Args:
                nums (List[int]): The input array of integers.

            Returns:
                List[int]: The prefix sum array.
            """
            prefix = [0] * len(nums)
            prefix[0] = nums[0]
            for i in range(1, len(nums)):
                prefix[i] = prefix[i - 1] + nums[i]
            return prefix

        def range_sum(self, left: int, right: int) -> int:
            """
            Calculates the sum of elements in the range [left, right] (inclusive).

            Args:
                left (int): Starting index of the range.
                right (int): Ending index of the range.

            Returns:
                int: Sum of elements in the specified range.

            Raises:
                ValueError: If the range indices are invalid.
            """
            if left < 0 or right >= len(self.prefix) or left > right:
                raise ValueError("Invalid range indices.")
            return self.prefix[right] - (self.prefix[left - 1] if left > 0 else 0)


    # Example for 1D Prefix Sum
    nums = [1, 2, 3, 4, 5]
    prefix_sum_1d = PrefixSum1D(nums)

    # Range sum from index 1 to 3 (i.e., 2 + 3 + 4)
    print(f"Range sum (1 to 3): {prefix_sum_1d.range_sum(1, 3)}")  # Output: 9


    class PrefixSum2D:
        def __init__(self, matrix: List[List[int]]):
            """
            Initializes the prefix sum for a 2D matrix.

            Args:
                matrix (List[List[int]]): The input 2D matrix of integers.

            Raises:
                ValueError: If the matrix is empty or contains empty rows.
            """
            if not matrix or not matrix[0]:
                raise ValueError("Input matrix must not be empty.")
            self.prefix = self._compute_prefix_sum_2d(matrix)

        @staticmethod
        def _compute_prefix_sum_2d(matrix: List[List[int]]) -> List[List[int]]:
            """
            Computes the prefix sum for the input 2D matrix.

            Args:
                matrix (List[List[int]]): The input 2D matrix of integers.

            Returns:
                List[List[int]]: The prefix sum 2D matrix.
            """
            rows, cols = len(matrix), len(matrix[0])
            prefix = [[0] * cols for _ in range(rows)]

            for i in range(rows):
                for j in range(cols):
                    prefix[i][j] = matrix[i][j]
                    if i > 0:
                        prefix[i][j] += prefix[i - 1][j]
                    if j > 0:
                        prefix[i][j] += prefix[i][j - 1]
                    if i > 0 and j > 0:
                        prefix[i][j] -= prefix[i - 1][j - 1]
            return prefix

        def range_sum(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> int:
            """
            Calculates the sum of elements within the submatrix defined by the given corners.

            Args:
                top_left (Tuple[int, int]): (row, col) for the top-left corner of the submatrix.
                bottom_right (Tuple[int, int]): (row, col) for the bottom-right corner of the submatrix.

            Returns:
                int: Sum of elements in the specified submatrix.

            Raises:
                ValueError: If the submatrix corners are invalid.
            """
            row1, col1 = top_left
            row2, col2 = bottom_right

            if row1 > row2 or col1 > col2:
                raise ValueError("Invalid submatrix corners.")

            total = self.prefix[row2][col2]
            if col1 > 0:
                total -= self.prefix[row2][col1 - 1]
            if row1 > 0:
                total -= self.prefix[row1 - 1][col2]
            if row1 > 0 and col1 > 0:
                total += self.prefix[row1 - 1][col1 - 1]

            return total


    # Example for 2D Prefix Sum
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    prefix_sum_2d = PrefixSum2D(matrix)

    # Sum of submatrix from (0, 0) to (1, 1) (top-left to bottom-right corner)
    print(f"Range sum (top-left (0,0) to bottom-right (1,1)): {prefix_sum_2d.range_sum((0, 0), (1, 1))}")  # Output: 12

    # Sum of submatrix from (1, 1) to (2, 2)
    print(f"Range sum (top-left (1,1) to bottom-right (2,2)): {prefix_sum_2d.range_sum((1, 1), (2, 2))}")  # Output: 28
    return


if __name__ == "__main__":
    app.run()
