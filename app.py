
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="ML Decision Platform",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# LOAD IRIS
# =====================================================

iris = load_iris()

iris_df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

iris_df["target"] = iris.target


# =====================================================
# LOAD DIABETES
# =====================================================

diabetes = load_diabetes()

diabetes_df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

diabetes_df["target"] = diabetes.target


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🤖 ML Decision Platform")

page = st.sidebar.selectbox(
    "Choose a page",
    [
        "Home",
        "Dataset",
        "Data Analysis",
        "Visualization",
        "Preprocessing",
        "Classification",
        "Regression",
        "Model Comparison",
        "Prediction"
    ]
)


# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.title("🤖 ML Decision Platform")

    st.write(
        "A simple Machine Learning platform for "
        "data analysis, visualization, model training "
        "and prediction."
    )

    st.subheader("Available Dataset Types")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Iris", "Classification")

    with col2:
        st.metric("Titanic", "Classification")

    with col3:
        st.metric("Diabetes", "Regression")

    with col4:
        st.metric("CSV", "Custom Data")

    st.subheader("Machine Learning Models")

    st.write("Classification:")
    st.write("• Logistic Regression")
    st.write("• KNN")
    st.write("• SVM")

    st.write("Regression:")
    st.write("• Linear Regression")
    st.write("• Multiple Linear Regression")

    st.subheader("Workflow")

    st.info(
        "Dataset → Analysis → Visualization → "
        "Preprocessing → Model Training → "
        "Comparison → Prediction"
    )


# =====================================================
# DATASET
# =====================================================

elif page == "Dataset":

    st.title("📂 Dataset Manager")

    dataset = st.selectbox(
        "Select Dataset",
        [
            "Iris",
            "Diabetes",
            "Upload CSV"
        ]
    )

    # -------------------------
    # IRIS
    # -------------------------

    if dataset == "Iris":

        selected_df = iris_df

        st.success("Iris dataset loaded successfully.")

        st.dataframe(selected_df)


    # -------------------------
    # DIABETES
    # -------------------------

    elif dataset == "Diabetes":

        selected_df = diabetes_df

        st.success(
            "Diabetes regression dataset loaded successfully."
        )

        st.dataframe(selected_df)


    # -------------------------
    # CSV
    # -------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:

            selected_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                "CSV uploaded successfully!"
            )

            st.dataframe(selected_df)

            st.write(
                "Rows:",
                selected_df.shape[0]
            )

            st.write(
                "Columns:",
                selected_df.shape[1]
            )


# =====================================================
# DATA ANALYSIS
# =====================================================

elif page == "Data Analysis":

    st.title("🔍 Data Analysis")

    dataset = st.selectbox(
        "Select Dataset",
        [
            "Iris",
            "Diabetes"
        ]
    )

    if dataset == "Iris":

        selected_df = iris_df

    else:

        selected_df = diabetes_df


    # Dataset information

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            selected_df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            selected_df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            selected_df.isnull().sum().sum()
        )

    with col4:

        st.metric(
            "Numerical Columns",
            len(
                selected_df.select_dtypes(
                    include="number"
                ).columns
            )
        )


    # Column analysis

    st.subheader("📊 Column Analysis")

    column = st.selectbox(
        "Select Column",
        selected_df.columns
    )

    if pd.api.types.is_numeric_dtype(
        selected_df[column]
    ):

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Mean",
                round(
                    selected_df[column].mean(),
                    2
                )
            )

        with col2:

            st.metric(
                "Median",
                round(
                    selected_df[column].median(),
                    2
                )
            )

        with col3:

            st.metric(
                "Minimum",
                round(
                    selected_df[column].min(),
                    2
                )
            )

        with col4:

            st.metric(
                "Maximum",
                round(
                    selected_df[column].max(),
                    2
                )
            )


    st.subheader("📈 Statistical Summary")

    st.dataframe(
        selected_df.describe()
    )


    st.subheader("⚠️ Missing Values")

    missing = pd.DataFrame({
        "Column": selected_df.columns,
        "Missing Values":
            selected_df.isnull().sum()
    })

    st.dataframe(missing)


# =====================================================
# VISUALIZATION
# =====================================================

elif page == "Visualization":

    st.title("📊 Data Visualization")

    dataset = st.selectbox(
        "Select Dataset",
        [
            "Iris",
            "Diabetes"
        ]
    )

    if dataset == "Iris":

        selected_df = iris_df

    else:

        selected_df = diabetes_df


    graph = st.selectbox(
        "Select Graph",
        [
            "Histogram",
            "Scatter Plot",
            "Box Plot"
        ]
    )


    if graph == "Histogram":

        column = st.selectbox(
            "Select Column",
            selected_df.columns
        )

        if st.button("Create Histogram"):

            fig, ax = plt.subplots()

            ax.hist(
                selected_df[column]
            )

            ax.set_xlabel(column)

            ax.set_ylabel(
                "Frequency"
            )

            ax.set_title(
                "Histogram"
            )

            st.pyplot(fig)


    elif graph == "Scatter Plot":

        x_column = st.selectbox(
            "X Axis",
            selected_df.columns
        )

        y_column = st.selectbox(
            "Y Axis",
            selected_df.columns
        )

        if st.button(
            "Create Scatter Plot"
        ):

            fig, ax = plt.subplots()

            ax.scatter(
                selected_df[x_column],
                selected_df[y_column]
            )

            ax.set_xlabel(
                x_column
            )

            ax.set_ylabel(
                y_column
            )

            st.pyplot(fig)


    else:

        column = st.selectbox(
            "Select Column",
            selected_df.columns
        )

        if st.button(
            "Create Box Plot"
        ):

            fig, ax = plt.subplots()

            ax.boxplot(
                selected_df[column]
            )

            ax.set_ylabel(
                column
            )

            st.pyplot(fig)


# =====================================================
# PREPROCESSING
# =====================================================

elif page == "Preprocessing":

    st.title("🧹 Data Preprocessing")

    dataset = st.selectbox(
        "Select Dataset",
        [
            "Iris",
            "Diabetes"
        ]
    )

    if dataset == "Iris":

        selected_df = iris_df

    else:

        selected_df = diabetes_df


    st.subheader(
        "Missing Value Check"
    )

    st.write(
        selected_df.isnull().sum()
    )


    st.subheader(
        "Feature Scaling"
    )

    scaling = st.checkbox(
        "Apply StandardScaler"
    )


    test_size = st.slider(
        "Test Size (%)",
        10,
        40,
        20
    )


    if st.button(
        "Prepare Data"
    ):

        X = selected_df.drop(
            "target",
            axis=1
        )

        y = selected_df["target"]


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42
        )


        if scaling:

            scaler = StandardScaler()

            X_train = scaler.fit_transform(
                X_train
            )

            X_test = scaler.transform(
                X_test
            )

            st.success(
                "StandardScaler applied successfully."
            )

        else:

            st.success(
                "Scaling was not applied."
            )


        st.write(
            "Training samples:",
            len(X_train)
        )

        st.write(
            "Testing samples:",
            len(X_test)
        )


# =====================================================
# CLASSIFICATION
# =====================================================

elif page == "Classification":

    st.title("🏷️ Classification")

    st.write(
        "Classification predicts a category."
    )


    dataset = st.selectbox(
        "Select Dataset",
        [
            "Iris"
        ]
    )


    model_name = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "KNN",
            "SVM"
        ]
    )


    X = iris.data

    y = iris.target


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    if model_name == "Logistic Regression":

        model = LogisticRegression(
            max_iter=1000
        )


    elif model_name == "KNN":

        neighbors = st.slider(
            "Number of Neighbors",
            1,
            15,
            5
        )

        model = KNeighborsClassifier(
            n_neighbors=neighbors
        )


    else:

        model = SVC()


    if st.button(
        "Train Model"
    ):

        model.fit(
            X_train,
            y_train
        )


        prediction = model.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            prediction
        )


        st.success(
            "Model trained successfully!"
        )


        st.metric(
            "Accuracy",
            str(
                round(
                    accuracy * 100,
                    2
                )
            ) + "%"
        )


        st.session_state[
            "classification_model"
        ] = model


# =====================================================
# REGRESSION
# =====================================================

elif page == "Regression":

    st.title("📉 Regression")

    st.write(
        "Regression predicts a numerical value."
    )


    model_type = st.selectbox(
        "Select Regression Model",
        [
            "Simple Linear Regression",
            "Multiple Linear Regression"
        ]
    )


    X = diabetes.data

    y = diabetes.target


    if model_type == "Simple Linear Regression":

        feature_number = st.slider(
            "Select Feature",
            0,
            9,
            0
        )

        X = X[
            :,
            feature_number
        ].reshape(
            -1,
            1
        )


    else:

        X = diabetes.data


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    if st.button(
        "Train Regression Model"
    ):

        model = LinearRegression()


        model.fit(
            X_train,
            y_train
        )


        prediction = model.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            prediction
        )


        mse = mean_squared_error(
            y_test,
            prediction
        )


        r2 = r2_score(
            y_test,
            prediction
        )


        st.success(
            "Regression model trained successfully!"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "MAE",
                round(mae, 2)
            )


        with col2:

            st.metric(
                "MSE",
                round(mse, 2)
            )


        with col3:

            st.metric(
                "R² Score",
                round(r2, 2)
            )


        st.session_state[
            "regression_model"
        ] = model


# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("🏆 Model Comparison")

    st.write(
        "Compare different classification models."
    )


    X = iris.data

    y = iris.target


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    if st.button(
        "Compare Models"
    ):


        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "KNN":
                KNeighborsClassifier(
                    n_neighbors=5
                ),

            "SVM":
                SVC()

        }


        results = []


        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )


            prediction = model.predict(
                X_test
            )


            accuracy = accuracy_score(
                y_test,
                prediction
            )


            results.append(
                [
                    name,
                    round(
                        accuracy * 100,
                        2
                    )
                ]
            )


        result_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Accuracy (%)"
            ]
        )


        st.dataframe(
            result_df
        )


        st.bar_chart(
            result_df.set_index(
                "Model"
            )
        )


        best_model = result_df.loc[
            result_df[
                "Accuracy (%)"
            ].idxmax()
        ]


        st.success(
            "Best Model: "
            + best_model["Model"]
        )


# =====================================================
# PREDICTION
# =====================================================

elif page == "Prediction":

    st.title("🔮 Iris Prediction")

    st.write(
        "Enter flower measurements."
    )


    sepal_length = st.number_input(
        "Sepal Length",
        0.0,
        10.0,
        5.1
    )


    sepal_width = st.number_input(
        "Sepal Width",
        0.0,
        10.0,
        3.5
    )


    petal_length = st.number_input(
        "Petal Length",
        0.0,
        10.0,
        1.4
    )


    petal_width = st.number_input(
        "Petal Width",
        0.0,
        10.0,
        0.2
    )


    model_name = st.selectbox(
        "Select Prediction Model",
        [
            "Logistic Regression",
            "KNN",
            "SVM"
        ]
    )


    if st.button(
        "Predict"
    ):


        if model_name == "Logistic Regression":

            model = LogisticRegression(
                max_iter=1000
            )


        elif model_name == "KNN":

            model = KNeighborsClassifier(
                n_neighbors=5
            )


        else:

            model = SVC()


        model.fit(
            iris.data,
            iris.target
        )


        new_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]


        prediction = model.predict(
            new_data
        )


        names = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]


        st.success(
            "Predicted Flower: "
            + names[
                prediction[0]
            ]
        )
