import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Backend Functions
def preprocess_data(data, target_column=None):
    """Splits the dataset into features and target, and further into train/test sets."""
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        # For clustering or other unsupervised learning
        X = data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

def train_model(X_train, y_train=None, model_type="regression", **kwargs):
    """Trains a model based on the specified type."""
    if model_type == "regression":
        model = LinearRegression()
    elif model_type == "classification":
        model = DecisionTreeClassifier()
    elif model_type == "clustering":
        n_clusters = kwargs.get("n_clusters", 3)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X_train)
        return model
    elif model_type == "neural_network":
        input_dim = X_train.shape[1]
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(32, activation='relu'),
            Dense(1, activation='linear' if kwargs.get("output_type") == "regression" else 'sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse' if kwargs.get("output_type") == "regression" else 'binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=kwargs.get("epochs", 10), batch_size=kwargs.get("batch_size", 32), verbose=1)
        return model
    else:
        raise ValueError("Invalid model type. Choose 'regression', 'classification', 'clustering', or 'neural_network'.")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test=None, model_type="regression", **kwargs):
    """Evaluates the trained model and returns metrics."""
    if model_type in ["regression", "classification"]:
        predictions = model.predict(X_test)
        if model_type == "regression":
            mse = mean_squared_error(y_test, predictions)
            return {"MSE": mse}
        elif model_type == "classification":
            acc = accuracy_score(y_test, predictions.round())
            conf_matrix = confusion_matrix(y_test, predictions.round())
            return {"Accuracy": acc, "Confusion Matrix": conf_matrix}
    elif model_type == "clustering":
        cluster_labels = model.predict(X_test)
        return {"Cluster Labels": cluster_labels}

# Streamlit Application
st.title("Enhanced Local Machine Learning Engine")

# Sidebar for file upload
st.sidebar.header("Step 1: Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV):")

if uploaded_file is not None:
    try:
        # Load dataset and display preview
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())

        # Sidebar: Choose target column or clustering
        st.sidebar.header("Step 2: Select Task")
        task_type = st.sidebar.selectbox("Choose Task Type", ["Regression", "Classification", "Clustering", "Neural Network"])

        if task_type in ["Regression", "Classification", "Neural Network"]:
            target_column = st.sidebar.selectbox("Choose the target column:", data.columns)

        # Preprocess data
        if st.sidebar.button("Preprocess Data"):
            if task_type == "Clustering":
                X_scaled = preprocess_data(data)
                st.session_state['X_scaled'] = X_scaled
                st.write("Data preprocessing complete for clustering!")
            else:
                X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
                st.session_state['X_train'], st.session_state['X_test'] = X_train, X_test
                st.session_state['y_train'], st.session_state['y_test'] = y_train, y_test
                st.write("Data preprocessing complete!")
                st.write("Training set size:", X_train.shape)
                st.write("Test set size:", X_test.shape)

        # Train Model
        if st.sidebar.button("Train Model"):
            if task_type == "Clustering":
                n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                model = train_model(st.session_state['X_scaled'], model_type="clustering", n_clusters=n_clusters)
                st.session_state['model'] = model
                st.write("Clustering model trained successfully!")
                st.write("Cluster Centers:")
                st.write(model.cluster_centers_)
            elif task_type == "Neural Network":
                epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100, value=10)
                batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=64, value=32)
                output_type = "regression" if task_type == "Regression" else "classification"
                model = train_model(st.session_state['X_train'], st.session_state['y_train'], model_type="neural_network", epochs=epochs, batch_size=batch_size, output_type=output_type)
                st.session_state['model'] = model
                st.write("Neural Network model trained successfully!")
            else:
                model_type = task_type.lower()
                model = train_model(st.session_state['X_train'], st.session_state['y_train'], model_type=model_type)
                st.session_state['model'] = model
                st.write(f"{task_type} model trained successfully!")

        # Evaluate Model
        if st.sidebar.button("Evaluate Model"):
            model_type = task_type.lower()
            if task_type == "Clustering":
                metrics = evaluate_model(st.session_state['model'], st.session_state['X_scaled'], model_type="clustering")
                st.write("### Clustering Results")
                st.write("Cluster Labels:")
                st.write(metrics["Cluster Labels"])
            elif task_type == "Neural Network":
                st.write("Evaluation for Neural Network currently relies on training history.")
            else:
                metrics = evaluate_model(st.session_state['model'], st.session_state['X_test'], st.session_state['y_test'], model_type=model_type)
                st.write("### Model Evaluation Results:")
                if task_type == "Regression":
                    st.write("Mean Squared Error (MSE):", metrics["MSE"])
                elif task_type == "Classification":
                    st.write("Accuracy:", metrics["Accuracy"])
                    st.write("Confusion Matrix:")
                    st.write(metrics["Confusion Matrix"])

                    # Plot confusion matrix
                    fig, ax = plt.subplots()
                    ax.matshow(metrics["Confusion Matrix"], cmap=plt.cm.Blues)
                    for (i, j), val in np.ndenumerate(metrics["Confusion Matrix"]):
                        ax.text(j, i, val, ha='center', va='center')
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    st.pyplot(fig)

        # Save the Model
        if st.sidebar.button("Save Model"):
            model_name = f"{task_type.lower()}_model.pkl"
            joblib.dump(st.session_state['model'], model_name)
            st.write(f"Model saved as {model_name}!")

    except Exception as e:
        st.error(f"Error: {e}")
