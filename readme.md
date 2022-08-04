# Landmark Recognition App
![Python](https://img.shields.io/badge/Python-0078D4?logo=python&logoColor=white)
![numpy](https://img.shields.io/badge/Numpy-777BB4?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-2C2D72?logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=Keras&logoColor=white)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?&logo=streamlit&logoColor=white)
![terminal](https://img.shields.io/badge/Windows%20Terminal-4D4D4D?&logo=Windows%20terminal&logoColor=white)
![vscode](https://img.shields.io/badge/Visual_Studio_Code-0078D4?&logo=visual%20studio%20code&logoColor=white)

Landmark Recognition App is used to predict landmark from user given image. it provides information about lankmark's full address, latitude & longitude and plot the predicted landmark on the map. The trained model [`landmarks_classifier_asia_V1/1`](https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1) is taken from the Tensorflow-Hub. There are total `98961` classes supported, in which Asia's most of the famous Landmark is covered.

## Installation
Open command prompt and create new environment
```
conda create -n your_env_name python = (any_version_number)
```
Then Activate the newly created environment
```
conda activate your_env_name
```
Clone the repository using `git`
```
git clone https://github.com/Prakashdeveloper03/Landmark-Recognition.git
```
Change to the cloned directory
```
cd <directory_name>
```
Then install all requirement packages for the app
```
pip install -r requirements.txt
```
Then, Run the `translator.py` script
```
streamlit run app.py
```
## 📷 Screenshots
### Brihadeeswara Temple
![sample1](images/s1.png)

### India Gate
![sample2](images/s2.png)