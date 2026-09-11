import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from importlib.metadata import version

    import dice_ml
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from dice_ml import Dice
    from sklearn.datasets import load_wine
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    return (
        DecisionTreeClassifier,
        Dice,
        Pipeline,
        StandardScaler,
        classification_report,
        confusion_matrix,
        dice_ml,
        load_wine,
        pd,
        plt,
        sns,
        train_test_split,
        version,
    )


@app.cell
def _(plt):
    plt.style.use("dark_background")
    return


@app.cell
def _(version):
    # Print package versions
    packages = ["dice_ml", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"]
    for package in packages:
        print(f"{package} version: {version(package)}")
    return


@app.cell
def _(load_wine):
    # Load and prepare data
    X, y = load_wine(as_frame=True, return_X_y=True)
    X = X.loc[:, ["alcohol", "flavanoids", "magnesium", "proline"]]
    X.columns = X.columns.str.capitalize()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | **Aspect**           | **Model Counterfactuals**                            | **Actual Counterfactuals**                          |
    |-----------------------|-----------------------------------------------------|----------------------------------------------------|
    | **Context**           | Within the bounds of a trained model.               | Concerned with the real world.                    |
    | **Objective**         | Explain or evaluate a model’s predictions.          | Understand true causal relationships or alternate realities. |
    | **Dependency**        | Depends on the model’s structure and assumptions.   | Requires a causal mechanism or data about the real world. |
    | **Use in AI/ML**      | Explainability, fairness, debugging, feature importance. | Causal inference, policy evaluation, treatment effects. |
    | **Example Question**  | "What if the input value of age were higher in the model?" | "What if the individual had gone to college?"     |
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    Pipeline,
    StandardScaler,
    X,
    train_test_split,
    y,
):
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    model_pipeline = Pipeline([("standardize", StandardScaler()), ("classify", DecisionTreeClassifier())])

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Model performance visualization
    y_pred = model_pipeline.predict(X_test)
    return X_test, model_pipeline, y_pred, y_test


@app.cell
def _(classification_report, confusion_matrix, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return (cm,)


@app.cell
def _(cm, plt, sns):
    plt.figure(figsize=(4, 3), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return


@app.cell
def _(Dice, X, X_test, dice_ml, model_pipeline, pd, y):
    # DiCE Counterfactual Analysis
    D = dice_ml.Data(
        dataframe=pd.concat([X, y], axis=1), continuous_features=X.columns.to_list(), outcome_name="target"
    )

    M = dice_ml.Model(model=model_pipeline, backend="sklearn", model_type="classifier")

    C = Dice(D, M, method="genetic")

    # Generate counterfactuals for a specific instance
    instance = X_test[4:5]
    y_pred_instance = model_pipeline.predict(instance)
    print("\nSelected Instance Prediction:", y_pred_instance)

    counterfactuals = C.generate_counterfactuals(instance, total_CFs=4, desired_class=2)

    # Visualize counterfactuals
    cf_df = counterfactuals.visualize_as_dataframe(show_only_changes=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1 style="font-weight: bold; background: linear-gradient(to right, teal, black); -webkit-background-clip: text; color: transparent;"> Model Counterfactuals vs. Actual Counterfactuals
    </h1>

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> 1. Model Counterfactuals
    </h2>

    ### Definition
    Hypothetical scenarios generated within the constraints of a trained model to explore alternative outcomes based on input changes.

    ### Objective
    To understand the model’s behavior and decision boundaries by asking:

    > What if $ X $ changes to $ X' $, how does $ Y $ (the output) change?

    ### Example
    Given a model $ f(X) $, we want to find the smallest perturbation $\Delta X$ such that:

    $
    f(X + \Delta X) = Y^*
    $

    where $ Y^* $ is the desired outcome or class.

    ---

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> 2. Actual Counterfactuals
    </h2>

    ### Framework
    - Causal inference techniques (e.g., Structural Causal Models, Potential Outcomes Framework).
    - Assumes a causal mechanism \( g(X) \) that relates inputs to real-world outcomes \( Y \).

    ### Key Property
    Actual counterfactuals of real-world events inherently deal with hypothetical alternate scenarios, which cannot be directly observed or verified because only the factual scenario is observable in our universe. Example:

    _*We cannot observe the same individual in two alternate realities simultaneously (e.g., as different genders).
    However, we can estimate or approximate potential outcomes by leveraging data about group-level effects (e.g., statistical or causal inferences drawn from populations).*_

    _*This ties into the core limitation of actual counterfactuals: they require assumptions and models (such as causal inference frameworks) to bridge the gap between the observed and unobserved scenarios. These assumptions influence the accuracy and reliability of the estimations.*_

    ### Applications
    - Policy evaluation (e.g., the effect of a treatment).
    - Understanding cause-effect relationships.

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> 3. DiCE Counterfactuals
    </h2>

    ### Definition
    Diverse Counterfactual Explanations (DiCE) focus on generating actionable, feasible, and diverse model counterfactuals.

    ### Objective
    To find minimal changes $ \Delta X $ in the input $ X $ such that the model’s prediction $ f(X + \Delta X) $ meets a desired condition:

    $$
    \text{Find } \Delta X \\
    \text{ minimizing } \| \Delta X \|_p \\
    \text{ subject to } f(X + \Delta X) = Y^*.
    $$

    ### Assumptions
    - The model $ f $ has learned a functional relationship approximating the true mapping between inputs and outputs.
    - Counterfactuals are not causal but operate within the model’s learned domain.

    ### Features
    - **Specifying which inputs to change**: Users can define which features are allowed to be altered.
    - **Difficulty of changing inputs**: DiCE allows assigning costs or difficulty levels to changing specific features.
    - **Specifying constraints**: Constraints such as bounds on feature values or interdependencies can be imposed to ensure realistic counterfactuals.
    """)
    return


if __name__ == "__main__":
    app.run()
