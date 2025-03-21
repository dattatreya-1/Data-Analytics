# -*- coding: utf-8 -*-
"""predstream.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iSLHomi0U6-xwUm-30PyInW4TOt6wPNG
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Software Reliability Prediction with ANN and Logistic Regression")

# Upload file in Streamlit
uploaded_file = st.file_uploader("promise_dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop unnecessary column if it exists
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    # Convert 'DL' column to integer binary values
    df['DL'] = df['DL'].astype(str)
    df['DL_extracted'] = np.where(df['DL'].str.contains('_TRUE', case=False, na=False), 1,
                                  np.where(df['DL'].str.contains('_FALSE', case=False) | df['DL'].str.contains('false', case=False), 0, -1))

    # Remove old column and rename
    df.drop('DL', axis=1, inplace=True)
    df.rename(columns={'DL_extracted': 'DL'}, inplace=True)

    # Display Data Preview
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Split into Features (X) and Target (y)
    X = df.drop(columns=['DL'])
    y = df['DL']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Logistic Regression Model
    log_reg = LogisticRegression(random_state=0)
    log_reg.fit(X_train, y_train)
    y_pred_LR = log_reg.predict(X_test)

    # Model Performance Metrics
    accuracy = accuracy_score(y_test, y_pred_LR)
    precision = precision_score(y_test, y_pred_LR)
    recall = recall_score(y_test, y_pred_LR)
    f1 = f1_score(y_test, y_pred_LR)

    # Display Metrics
    st.write("### Logistic Regression Performance")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    # Compute percentiles for fuzzy membership classification
    selected_features = ['maxNUM_UNIQUE_OPERANDS', 'maxNUM_UNIQUE_OPERATORS',
                         'COUPLING_BETWEEN_OBJECTS', 'maxHALSTEAD_DIFFICULTY', 'maxNUM_OPERANDS']

    low_threshold = df[selected_features].quantile(0.33)
    medium_threshold = df[selected_features].quantile(0.66)

    # Define fuzzy membership function
    def fuzzy_membership(value, feature):
        if value <= low_threshold.get(feature, np.inf):
            return 0.2
        elif value <= medium_threshold.get(feature, np.inf):
            return 0.5
        else:
            return 0.8

    # Apply transformation
    df['refined_fuzzy_defect_likelihood'] = df.apply(
        lambda row: np.mean([fuzzy_membership(row[feature], feature) for feature in selected_features]), axis=1)

    # Add refined feature to training data
    X_train['refined_fuzzy_defect_likelihood'] = df.loc[X_train.index, 'refined_fuzzy_defect_likelihood']
    X_test['refined_fuzzy_defect_likelihood'] = df.loc[X_test.index, 'refined_fuzzy_defect_likelihood']

    # ANN Model
    ann_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = ann_model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test),
                            verbose=0, callbacks=[early_stopping])

    # ANN Predictions
    y_pred_ANN_refined = (ann_model.predict(X_test) > 0.5).astype(int).flatten()

    # ANN Performance
    accuracy_ann_refined = accuracy_score(y_test, y_pred_ANN_refined)
    precision_ann_refined = precision_score(y_test, y_pred_ANN_refined)
    recall_ann_refined = recall_score(y_test, y_pred_ANN_refined)
    f1_ann_refined = f1_score(y_test, y_pred_ANN_refined)

    # Display ANN Metrics
    st.write("### ANN Model Performance")
    st.write(f"Accuracy: {accuracy_ann_refined:.4f}")
    st.write(f"Precision: {precision_ann_refined:.4f}")
    st.write(f"Recall: {recall_ann_refined:.4f}")
    st.write(f"F1 Score: {f1_ann_refined:.4f}")

    # Save Model
    ann_model.save('model2.h5')
    st.success("Model saved as 'model2.h5'")