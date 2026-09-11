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
    ### Distance between two parallel hyperplanes
    1. **Hyperplane**:
    A hyperplane in $\mathbb{R}^n$ is defined by the equation:
    $$
    \mathbf{w} \cdot \mathbf{x} + b = 0
    $$
    where $\mathbf{w}$ is the normal vector to the hyperplane, $\mathbf{x}$ is a vector on the hyperplane, and $b$ is a constant.

    2. **Parallel Hyperplanes**:
    Two hyperplanes are parallel if their normal vectors are the same or proportional. For hyperplanes:
    $$
    \mathbf{w} \cdot \mathbf{x} + b_1 = 0 \quad \text{and} \quad \mathbf{w} \cdot \mathbf{x} + b_2 = 0
    $$
    They are parallel if $\mathbf{w}_1 = \mathbf{w}_2$.

    3. **Distance Between Two Vectors**:
    The distance between two vectors $\mathbf{v}_1$ and $\mathbf{v}_2$ in $\mathbb{R}^n$ is given by:
    $$
    d = \|\mathbf{v}_1 - \mathbf{v}_2\|
    $$

    4. **Projection of a Vector onto another Vector**:
    The projection of vector $\mathbf{v}_1$ onto vector $\mathbf{v}_2$ is:
    $$
    \text{proj}_{\mathbf{v}_2} \mathbf{v}_1 = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_2\|^2} \mathbf{v}_2
    $$

    5. **Scalar Projection (Component of a vector along another vector)**:
    The scalar projection of vector $\mathbf{v}_1$ onto vector $\mathbf{v}_2$ is:
    $$
    \text{comp}_{\mathbf{v}_2} \mathbf{v}_1 = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_2\|}
    $$

    6. **Distance Between Two Parallel Hyperplanes**:
    The distance $D$ between two parallel hyperplanes $\mathbf{w} \cdot \mathbf{x} + b_1 = 0$ and $\mathbf{w} \cdot \mathbf{x} + b_2 = 0$ is:
    $$
    D = \frac{|b_2 - b_1|}{\|\mathbf{w}\|}
    $$
    where $\mathbf{w}$ is the normal vector to the hyperplanes, and $b_1 \neq b_2$.
    ---
    ### Example: Distance Between Two Parallel Hyperplanes
    Given:
    - $ P_1: \mathbf{w} \cdot \mathbf{x} + b_1 = 0 $ where $\mathbf{w} = (2, 3, 4)$ and $b_1 = 0$
    - $ P_2: \mathbf{w} \cdot \mathbf{x} + b_2 = 0 $ where $\mathbf{w} = (2, 3, 4)$ and $b_2 = 10$

    **Step 1: Identify the normal vector**
    The normal vector $\mathbf{w}$ is the same for both hyperplanes:
    $$
    \mathbf{w} = (2, 3, 4)
    $$

    **Step 2: Extract constants $b_1$ and $b_2$**
    From the equations:
    - For $P_1$, $b_1 = 0$.
    - For $P_2$, $b_2 = 10$.

    **Step 3: Apply the distance formula**
    The distance $D$ between the two parallel hyperplanes is:
    $$
    D = \frac{|b_2 - b_1|}{\|\mathbf{w}\|}
    $$
    Calculate the norm of $\mathbf{w}$:
    $$
    \|\mathbf{w}\| = \sqrt{2^2 + 3^2 + 4^2} = \sqrt{4 + 9 + 16} = \sqrt{29}
    $$
    Now, substitute the values into the formula:
    $$
    D = \frac{|10 - 0|}{\sqrt{29}} = \frac{10}{\sqrt{29}}
    $$

    **Step 4: Final Answer**
    $$
    D = \frac{10}{\sqrt{29}} \approx 1.86
    $$
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    return np, plt


@app.cell
def _(plt):
    plt.style.use('dark_background')
    plt.rc('axes', titlesize=24, labelsize=20, labelpad=5)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)              
    plt.rc('legend', fontsize=12)
    return


@app.cell
def _(np):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    x, y = np.meshgrid(x, y)
    return x, y


@app.cell
def _(x, y):
    # equation-1: 2x + 3y + 4z = 0 
    # z = -(2x + 3y) / 4
    z1 = -(2*x + 3*y) / 4
    # equation-2: 2x + 3y + 4z - 10 
    # z = -(2x + 3y - 10) / 4
    z2 = -(2 * x + 3 * y - 10) / 4
    return z1, z2


@app.cell
def _(np, plt, x, y, z1, z2):
    # Create the figure
    fig = plt.figure(figsize=(15, 12), dpi=300)

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.view_init(elev=30, azim=30)
    ax1.plot_surface(x, y, z1, alpha=0.8, cmap="Blues", edgecolor="k", 
                     rstride=10, cstride=10, label='P1: 2x + 3y + 4z = 0')
    ax1.plot_surface(x, y, z2, alpha=0.8, cmap="Reds", edgecolor="k", 
                     rstride=10, cstride=10, label='P2: 2x + 3y + 4z - 10 = 0')
    ax1.set_title("3D Plot", pad=20)
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    ax1.legend(loc='upper right')
    # Plot 2: XY projection
    ax2 = fig.add_subplot(222)
    ax2.contourf(x, y, z1, levels=50, cmap="Blues", alpha=0.4)
    ax2.contourf(x, y, z2, levels=50, cmap="Reds", alpha=0.3)
    ax2.set_title("XY Projection")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.grid(True)

    # Plot 3: XZ projection
    ax3 = fig.add_subplot(223)
    x_line = np.linspace(-10, 10, 100)
    z1_line = -(2 * x_line + 3 * 0) / 4
    z2_line = -(2 * x_line + 3 * 0 - 10) / 4
    ax3.plot(x_line, z1_line, color="blue", label='P1: 2x + 3y + 4z = 0')
    ax3.plot(x_line, z2_line, label='P2: 2x + 3y + 4z - 10 = 0', color="red")
    ax3.set_title("XZ Projection")
    ax3.set_xlabel("X-axis")
    ax3.set_ylabel("Z-axis")
    ax3.grid(True)
    ax3.legend()

    # Plot 4: YZ projection
    ax4 = fig.add_subplot(224)
    y_line = np.linspace(-10, 10, 100)
    z1_line_y = -(2 * 0 + 3 * y_line) / 4
    z2_line_y = -(2 * 0 + 3 * y_line - 10) / 4
    ax4.plot(y_line, z1_line_y, label='P1: 2x + 3y + 4z = 0', color="blue")
    ax4.plot(y_line, z2_line_y, label='P2: 2x + 3y + 4z - 10 = 0', color="red")
    ax4.set_title("YZ Projection")
    ax4.set_xlabel("Y-axis")
    ax4.set_ylabel("Z-axis")
    ax4.grid(True)
    ax4.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    answer = 1.8569
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
