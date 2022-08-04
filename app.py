import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import Image
from geopy.geocoders import Nominatim

# setting app's title, icon & layout
st.set_page_config(page_title="landmark recognition", page_icon="🎯")

# css style to hide footer, header and main menu details
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# asia landmarks classifier model version 1 url link
model_url = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1"

# reads "landmarks_classifier_asia_V1_label_map.csv" as a pandas dataframe
df = pd.read_csv("data/landmarks_classifier_asia_V1_label_map.csv")

# creates a dict object by using dataframe id column as key and name column as values
labels = dict(zip(df.id, df.name))

# stores the classification model
classifier = tf.keras.Sequential(
    [
        hub.KerasLayer(
            model_url, input_shape=(321, 321) + (3,), output_key="predictions:logits"
        )
    ]
)


def image_processing(image):
    original_img = Image.open(image)  # opens & stores passed image object
    original_img = original_img.resize((321, 321))  # resizes image shape to 321 x 321
    # by dividing by 255, the 0-255 range can be described with a 0.0-1.0 range
    original_img = (
        np.array(original_img) / 255.0
    )  # where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF)
    original_img = original_img[
        np.newaxis
    ]  # used to increase the dimension of the existing array by one more dimension, when used once
    result = classifier.predict(
        original_img
    )  # predicts the landmark using the classifier model
    # returns index of the maximum prediction value & resized image
    return labels[np.argmax(result)]


def get_map(loc):
    geolocator = Nominatim(
        user_agent="Landmark"
    )  # Nominatim geocoder for OpenStreetMap data
    location = geolocator.geocode(loc)  # Return a location point by address.
    # returns given location's address, latitude and longitude
    return location.address, location.latitude, location.longitude


def main():
    st.header("Landmark Recognition")  # sets header text
    # get image file from user by file_uploader
    uploaded_file = st.file_uploader("Choose your Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)
        with st.spinner("Predicting..."):
            prediction = image_processing(
                uploaded_file
            )  # calls image_processing() to predict

        # displays the predicted text
        st.success(f"Predicted Landmark is: {prediction.capitalize()}")
        try:
            address, latitude, longitude = get_map(
                prediction
            )  # get location's address, latitude and longitude details
            # shows the address of the location
            with st.expander(
                f"Address details of {prediction.capitalize()}", expanded=True
            ):
                st.text(address.replace(", ", ",\n"))
            loc_dict = {"Latitude": latitude, "Longitude": longitude}
            # shows the address of the location's latitude and longitude
            with st.expander(
                f"Latitude & Longitude of {prediction.capitalize()}", expanded=True
            ):
                st.json(loc_dict)
            data = [[latitude, longitude]]
            df = pd.DataFrame(data, columns=["lat", "lon"])
            # displays the map of the location using latitude and longitude
            with st.expander(f"{prediction.capitalize()} on the Map 🗺️"):
                st.map(df)
        except Exception as e:
            st.warning("No address found!!")


if __name__ == "__main__":
    main()  # calls the main()
