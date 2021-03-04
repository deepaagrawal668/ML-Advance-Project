import pickle
import streamlit as st
#from PIL import Image

model = pickle.load(open('random_forest_model.pkl', 'rb'))


def main():
    html_temp = """
    <div style = "background color: #5F4888FF; padding:10px">
    <h1 style="color:#FFFF;">Breast Cancer Prediction</h1>
    </div>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['About the model', 'About Prediction']
    option = st.sidebar.selectbox('Menu', activities)

    if option == 'About the model':
         html_temp_about = """
         The score of the model is 97.20%
         
         It takes 16 parameters so read the heading properly and then write the output in integer manner to get the 
         proper output.
         """

         st.sidebar.markdown(html_temp_about, unsafe_allow_html=True)

    elif option == 'About Prediction':
        html_temp_prediction = """
        Malignant: -
        If Prediction is Malignant that means you are suffering from breast cancer.
        Malignant refers to cancer cells that can invade and kill nearby tissue and spread to other parts of your body.
        
        Benign: -
        If Prediction is Benign that means you don't have cancer.
        Benign refers to a condition, tumor, or growth that is not cancerous. 
        This means that it does not spread to other parts of the body
        
        """

        st.sidebar.markdown(html_temp_prediction, unsafe_allow_html=True)

    radius_mean = float(st.text_input("Mean of the Radius", "e.g 17.99"))
    texture_mean = float(st.text_input("Mean of the Texture", "e.g 10.38"))
    perimeter_mean = float(st.text_input("Mean of the Perimeter", "e.g 122.80"))
    area_mean = float(st.text_input("Mean of the Area", "e.g 1001.0	"))
    concavity_mean = float(st.text_input("Mean of the Concavity", "e.g 0.3001"))
    concave_points_mean = float(st.text_input("Mean of the Concave points", "e.g 0.14710"))
    radius_se = float(st.text_input("Radius se", "e.g 1.0950"))
    perimeter_se = float(st.text_input("Perimeter se", "e.g 8.589"))
    area_se = float(st.text_input("Area se", "e.g 153.40"))
    radius_worst = float(st.text_input("Radius worst", "e.g 25.38"))
    texture_worst = float(st.text_input("Texture worst", "e.g 17.33"))
    perimeter_worst = float(st.text_input("Perimeter worst", "e.g 184.60"))
    area_worst = float(st.text_input("Area worst", "e.g 2019.0"))
    compactness_worst = float(st.text_input("Compactness worst", "e.g 0.6656"))
    concavity_worst = float(st.text_input("Concavity worst", "e.g 0.7119"))
    concave_points_worst = float(st.text_input("Concave points worst", "e.g 0.2654"))


    inputs = [[radius_mean, texture_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean,
               radius_se, perimeter_se, area_se, radius_worst, texture_worst,
                perimeter_worst, area_worst, compactness_worst, concavity_worst,
                concave_points_worst]]

    st.subheader("Prediction")

    if st.button("Predict"):
        st.write("**Predicted Result**")
        a = model.predict(inputs)[0]
        if a == 0:
            st.success("Benign")
        elif a == 1:
            st.error("Malignant")
    else:
        pass




main()