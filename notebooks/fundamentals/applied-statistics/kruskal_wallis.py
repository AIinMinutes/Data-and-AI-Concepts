import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Kruskal-Wallis Test
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 
    import statsmodels.api as sm
    from scipy.stats import shapiro, levene, kruskal

    np.random.seed(47); plt.style.use('dark_background')

    # Let's generate data: 3 groups with unequal means
    group1, group2, group3 = map(
        lambda loc: np.random.laplace(loc=loc, scale=5, size=100), 
        [20, 21, 22]
    )
    data = pd.DataFrame({
        'Value': np.concatenate([group1, group2, group3]),
        'Group': np.repeat(['Group 1', 'Group 2', 'Group 3'], 
                           [group1.size, group2.size, group3.size]
                          )
    })
    data.sample(3, random_state=4)
    return data, group1, group2, group3, kruskal, levene, plt, shapiro, sm, sns


@app.cell
def _(data, plt, sns):
    # Homoscedasticity (equal variance)
    plt.style.use('dark_background')
    plt.figure(figsize=(3, 2), dpi=150)
    sns.violinplot(x='Group', y='Value', data=data, hue='Group', 
                   palette=['red', 'green', 'blue'])
    plt.title('Boxplot for Each Group')
    plt.xlabel('Group')
    plt.ylabel('Value')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **Levene's test** is a statistical test used to assess the **equality of variances** across different groups by evaluating the **absolute deviations** from the group means (or medians).
    <br> <br>
    If the p-value is less than **the set significance level (0.05 usually)**, it indicates significant evidence to reject the null hypothesis, suggesting that the **variances are not equal** across the groups.
    """)
    return


@app.cell
def _(group1, group2, group3, levene):
    # Levene’s Test
    _stat, _p = levene(group1, group2, group3)
    print(f'Levene’s Test for Equality of Variances: \nW-statistic={_stat:.4f}, p-value={_p:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Conclusion-1:** We failed to reject the null hypothesis that variances are equal. The samples are homoscedastic.
    """)
    return


@app.cell
def _(group1, group2, group3, plt, sm, sns):
    gs, cs = ([group1, group2, group3], ['r', 'g', 'b'])
    ths = [f'Group {_i + 1} Histogram' for _i in range(3)]
    tqs = [f'Group {_i + 1} Q-Q Plot' for _i in range(3)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for _i, (g, c, h, q) in enumerate(zip(gs, cs, ths, tqs)):
        sns.histplot(g, kde=True, ax=axes[0, _i], color=c)
        sm.qqplot(g, line='s', ax=axes[1, _i], markerfacecolor=c)
        axes[0, _i].set_title(h)
        axes[1, _i].set_title(q)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **Shapiro-Wilk test** is a statistical test used to assess the **normality** of a dataset by comparing the observed distribution of data to a theoretical normal distribution. If the **p-value** is less than **the level of significance (0.05 usually)**, it indicates significant evidence to reject the **null hypothesis**, suggesting that the data does not follow a normal distribution.
    """)
    return


@app.cell
def _(group1, group2, group3, shapiro):
    # Shapiro-Wilk Test for normality
    print('Shapiro-Wilk Test Results:')
    for _i, group in enumerate([group1, group2, group3], start=1):
        _stat, _p = shapiro(group)
        print(f'Group {_i}: W-statistic={_stat:.4f}, p-value={_p:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Conclusion-2:** None of the groups' population is normally distributed.
    """)
    return


@app.cell
def _(group1, group2, group3, kruskal):
    _stat, p_value = kruskal(group1, group2, group3)
    print(f'Kruskal-Wallis H-statistic: {_stat:.4f} \np-value: {p_value:.4f}')
    if p_value < 0.05:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Final Conclusion
    We generated data from a **Laplace distribution** (double exponential). **Levene's test** indicated equal variances across groups. The **Shapiro-Wilk test** confirmed that the samples do not follow a normal distribution. As the normality assumption for **ANOVA** is violated, we used the **Kruskal-Wallis test**, a non-parametric alternative to ANOVA. It shows we have sufficient efficience to reject the null hypothesis that is the medians of the three groups are equal.
    """)
    return


if __name__ == "__main__":
    app.run()
