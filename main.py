import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title of the app
st.title("Linear Regression App")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the data
    st.write("Data from CSV:")
    st.dataframe(data)

    # Check if the required columns are present
    if 'X' in data.columns and 'Y' in data.columns:
        # Get X and Y values from the CSV
        X = data[['X']].values
        Y = data['Y'].values

        # Create a linear regression model
        model = LinearRegression()
        model.fit(X, Y)

        # Display the coefficients
        st.write(f"Coefficient (slope): {model.coef_[0]}")
        st.write(f"Intercept: {model.intercept_}")

        # User input for additional X and Y values
        st.subheader("Add your own X and Y values:")
        user_x = st.number_input("Enter X value:", value=0.0)
        user_y = st.number_input("Enter Y value:", value=0.0)

        # Predict Y for the user input X
        predicted_y = model.predict(np.array([[user_x]]))
        st.write(f"Predicted Y for X={user_x}: {predicted_y[0]}")

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, color='blue', label='Data Points')
        plt.plot(X, model.predict(X), color='red', label='Regression Line')
        plt.scatter(user_x, user_y, color='green', label='User  Input', marker='x', s=200)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        st.pyplot(plt)

    else:
        st.error("The uploaded CSV file must contain 'X' and 'Y' columns.")
