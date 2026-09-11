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
    # Understanding Einstein Summation Notation in NumPy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Concept
    With `einsum`, you:
    1. Define labels for each axis of your tensors
    2. Specify which labels appear in the output
    3. Repeated labels indicate summation along those axes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dot Product
    """)
    return


@app.cell
def _(np):
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    # Regular method
    dot_product_regular = np.dot(v1, v2)
    print("Dot product using np.dot():", dot_product_regular)

    # Using einsum: repeated index 'i' means sum over this dimension
    dot_product_einsum = np.einsum('i,i->', v1, v2)
    print("Dot product using einsum:", dot_product_einsum)
    return v1, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Outer Product
    """)
    return


@app.cell
def _(np, v1, v2):
    # Regular method
    outer_product_regular = np.outer(v1, v2)
    print("Outer product using np.outer():\n", outer_product_regular)

    # Using einsum: separate indices 'i,j' with no summation
    outer_product_einsum = np.einsum('i,j->ij', v1, v2)
    print("Outer product using einsum:\n", outer_product_einsum)
    return


@app.cell
def _(np):
    ### Matrix-Vector Multiplication
    _A = np.array([[4, 5], [6, 7]])
    v = np.array([4, 5])
    mv_regular = _A @ v
    # Regular method
    print('Matrix-vector product using @:\n', mv_regular)  # or np.matmul(A, v)
    mv_einsum = np.einsum('ij,j->i', _A, v)
    # Using einsum: sum over the repeated index 'j'
    print('Matrix-vector product using einsum:\n', mv_einsum)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matrix-Matrix Multiplication
    """)
    return


@app.cell
def _(np):
    _A = np.array([[4, 5], [6, 7]])
    _B = np.array([[3, 4, 5], [6, 8, 1]]).reshape(2, 3)
    mm_regular = _A @ _B
    # Regular method
    print('Matrix-matrix product using @:\n', mm_regular)  # or np.matmul(A, B)
    mm_einsum = np.einsum('ij,jk->ik', _A, _B)
    # Using einsum: sum over the repeated index 'j'
    print('Matrix-matrix product using einsum:\n', mm_einsum)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Element-wise Multiplication
    """)
    return


@app.cell
def _(np):
    _A = np.array([[3, 1], [3, 4]])
    _B = np.array([[4, 5], [1, 2]])
    elementwise_regular = _A * _B
    # Regular method
    print('Element-wise product using *:\n', elementwise_regular)
    elementwise_einsum = np.einsum('ij,ij->ij', _A, _B)
    # Using einsum: indices remain the same for input and output
    print('Element-wise product using einsum:\n', elementwise_einsum)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matrix Multiplication Batch-wise
    """)
    return


@app.cell
def _(np):
    # 3D example
    _X = np.random.rand(2, 3, 4)  # Shape (2, 3, 4)
    Y = np.random.rand(2, 4, 5)  # Shape (2, 4, 5)
    result = np.einsum('brc,bct->brt', _X, Y)
    # Let's interpret these as:
    # X: 2 batches, 3 rows, 4 columns
    # Y: 2 batches, 4 rows, 5 columns
    print('X shape:', _X.shape)
    # Batch matrix multiplication: brc,bct->brt
    # Sum over c (columns of X, rows of Y)
    # Free indices: b (batch), r (rows of X), t (columns of Y)
    print('Y shape:', Y.shape)
    print('Result shape:', result.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Permuting Axes
    """)
    return


@app.cell
def _(np):
    # Transpose a matrix
    _A = np.array([[1, 2, 3], [4, 5, 6]])
    print('Original A:\n', _A)
    print('A transpose using .T:\n', _A.T)
    print('A transpose using einsum:\n', np.einsum('ij->ji', _A))
    _X = np.random.rand(2, 3, 4)
    # Permute axes of a 3D tensor
    print('Original X shape:', _X.shape)
    print('Permuted X shape:', np.einsum('ijk->kji', _X).shape)  # Should be (4, 3, 2)
    return


if __name__ == "__main__":
    app.run()
