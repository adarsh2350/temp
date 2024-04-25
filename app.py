import streamlit as st
import joblib
import numpy as np



# Load the pre-trained model
model = joblib.load("rf.joblib")

# Define a function to predict the experimental class
def predict_class(params):
    # Preprocess the input parameters if necessary
    # Make sure the input params are in the same order as when the model was trained
    # You may need to convert the input into a numpy array or DataFrame depending on your model

    # Predict the class
    predicted_class = model.predict(params.reshape(1, -1))[0]
    return predicted_class

# Streamlit app
def main():
    # Title of the app
    st.title("Experimental Class Prediction")

    # Input fields for parameters
    st.sidebar.title("Input Parameters")
    parameters = []
    for param in ['J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC', 'nCb-', 'C%', 'nCp', 'nO', 'F03[C-N]', 
                  'SdssC', 'HyWi_B(m)', 'LOC', 'F03[C-O]', 'Mi', 'nN-N', 'nArNO2', 'nCRX3', 'SpPosA_B(p)', 
                  'nCIR', 'B01[C-Br]', 'B03[C-Cl]', 'N-073', 'SpMax_A', 'Psi_i_1d', 'B04[C-Br]', 'SdO', 'TI2_L', 'nCrt', 
                  'C-026', 'F02[C-N]', 'nHDon', 'Psi_i_A', 'nN', 'SM6_B(m)', 'nArCOOR', 'nX']:
        parameters.append(st.sidebar.number_input(param, step=0.01))

    # Predict button
    if st.sidebar.button("Predict"):
        # Convert parameters to numpy array
        params_array = np.array(parameters)
        # Predict the class
        predicted_class = predict_class(params_array)
        # Display the predicted class
        if predict_class == 0:
            st.write(f"Predicted Experimental Class: RB")
        else:
            st.write(f"Predicted Experimental Class: NRB")

if __name__ == "__main__":
    main()
