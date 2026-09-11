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
    ### Gradient

    - **Definition**: The gradient of a scalar-valued function $f(\textbf{x})$ is a column vector of partial derivatives with respect to each element of a column vector $\textbf{x}$.

    - **Mathematical Expression**: If $f(\textbf{x})$ is a scalar function of $\textbf{x} = \begin{bmatrix} x_1 & x_2 & \dots & x_n \end{bmatrix}^T$, then the gradient is:
      $$
      \nabla f(\textbf{x}) = \begin{bmatrix}
      \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \dots & \frac{\partial f}{\partial x_n}
      \end{bmatrix}^T
      $$

    **Example**: If $f(x, y) = x^2 + 3xy$, then
    $$
    \nabla f(x, y) = \begin{bmatrix}
    \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y}
    \end{bmatrix}^T
    = \begin{bmatrix}
    2x + 3y & 3x
    \end{bmatrix}^T
    $$
    This is the gradient of the scalar function $f(x, y)$, representing the rate of change with respect to each variable and pointing in the direction of the steepest ascent at $(x, y)$.

    ### Jacobian

    - **Definition**: The Jacobian is a matrix of all first-order partial derivatives of a vector-valued function.

    - **Mathematical Expression**: For a vector-valued function $\mathbf{y}(\textbf{x}) = \begin{bmatrix} y_1 & y_2 \end{bmatrix}^T$, where $\mathbf{x} = \begin{bmatrix} x_1 & x_2 \end{bmatrix}^T$, the Jacobian matrix is:
      $$
      J = \begin{bmatrix}
      \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\
      \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2}
      \end{bmatrix}
      $$

    **Example**: If $\mathbf{y}(x, y) = \begin{bmatrix} x^2 + 3xy & 2x + y \end{bmatrix}^T$, the Jacobian matrix is:
    $$
    J = \begin{bmatrix}
    \frac{\partial (x^2 + 3xy)}{\partial x} & \frac{\partial (x^2 + 3xy)}{\partial y} \\
    \frac{\partial (2x + y)}{\partial x} & \frac{\partial (2x + y)}{\partial y}
    \end{bmatrix}
    \hspace{1cm}
    J = \begin{bmatrix}
    2x + 3y & 3x \\
    2 & 1
    \end{bmatrix}
    $$

    This **2 x 2 matrix** represents the rate of change of each component of the output vector $\mathbf{y}$ with respect to each input variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top-5 Matrix Calculus Rules ###

    ### Rule-1 ###

    Given a function $f(x) = a^T x$, where:
    - $a$ is a $n \times 1$ vector,
    - $x$ is a $n \times 1$ vector,

    the gradient of $f(x)$ with respect to $x$ is:

    $$
    \nabla_x f = a
    $$
    """)
    return


@app.cell
def _():
    import torch
    torch.manual_seed(47)
    _a = torch.randn(2, 1)
    _x = torch.randn(2, 1, requires_grad=True)

    def _grad_f(x, a):
        f = _a.T @ _x
        f.backward()
        return _x.grad
    _expected_gradient = _a
    _calculated_gradient = _grad_f(_x, _a)
    assert torch.allclose(_expected_gradient, _calculated_gradient)
    print(_calculated_gradient.tolist())
    return (torch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rule-2 ###

    Given a function $ f(x) = A x $, where:
    - $ A $ is an $ m \times n $ matrix,
    - $ x $ is an $ n \times 1 $ vector,

    the Jacobian of $ f(x) $ with respect to $ x $ is:

    $$
    \mathbf{J}_{f(x)} = A
    $$
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(47)
    _A = torch.randn(2, 3)
    _x = torch.randn(3, 1, requires_grad=True)

    def f(x):
        return _A @ _x
    jacobian = torch.autograd.functional.jacobian(f, _x).reshape(2, -1)
    expected_jacobian = _A
    assert torch.allclose(jacobian, expected_jacobian)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rule-3

    Given a function $f(x) = x^T A x$, where:
    - $A$ is a $n \times n$ matrix,
    - $x$ is a $n \times 1$ vector,

    the gradient of $f(x)$ with respect to $x$ is:

    $$
    \nabla_x f = A x + A^T x
    $$

    #### Condition on $A$:
    - If $A$ is **symmetric** ($A = A^T$), the gradient simplifies to, $\nabla_x f = 2 A x$
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(47)
    _A = torch.randn(2, 2)
    _x = torch.randn(2, 1, requires_grad=True)

    def _grad_f(A, x):
        f = _x.T @ _A @ _x
        f.backward()
        return _x.grad
    _expected_gradient = _A @ _x + _A.T @ _x
    _calculated_gradient = _grad_f(_A, _x)
    assert torch.allclose(_expected_gradient, _calculated_gradient)
    print(_calculated_gradient.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rule-4 ###

    Given a function $f(x, y) = x^T A y$, where:
    - $A$ is a $n \times n$ matrix,
    - $x$ is a $n \times 1$ vector,
    - $y$ is a $n \times 1$ vector,

    the gradients of $f(x, y)$ with respect to $x$ and $y$ are:

    $$
    \nabla_x f = A y
    $$

    $$
    \nabla_y f = A^T x
    $$
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(47)
    _A = torch.randn(2, 2)
    _x = torch.randn(2, 1, requires_grad=True)
    y = torch.randn(2, 1, requires_grad=True)

    def _grad_f(A, x, y):
        f = _x.T @ _A @ y
        f.backward()
        return (_x.grad, y.grad)
    expected_grad_x = _A @ y
    expected_grad_y = _A.T @ _x
    calculated_grad_x, calculated_grad_y = _grad_f(_A, _x, y)
    assert torch.allclose(expected_grad_x, calculated_grad_x)
    assert torch.allclose(expected_grad_y, calculated_grad_y)
    print('Calculated Gradient with respect to y:')
    print(calculated_grad_y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rule-5

    Given a function $ f(X) = a^T X b $, where:
    - $ a $ is a $n \times 1 $ column vector,
    - $ X $ is a $n \times m $ matrix,
    - $ b $ is a $m \times 1 $ column vector,

    the gradient of $ f(X) $ with respect to $X $ is:

    $$
    \nabla_X (a^T X b) = a b^T
    $$
    """)
    return


@app.cell
def _(X, torch):
    torch.manual_seed(47)
    _a = torch.randn(3, 1)
    b = torch.randn(2, 1)
    zX = torch.randn(3, 2, requires_grad=True)

    def _grad_f(X, a, b):
        f = _a.T @ X @ b
        f.backward()
        return X.grad
    calculated_grad_X = _grad_f(X, _a, b)
    expected_grad_X = _a @ b.T
    assert torch.allclose(expected_grad_X, calculated_grad_X)
    print(calculated_grad_X)
    return


if __name__ == "__main__":
    app.run()
