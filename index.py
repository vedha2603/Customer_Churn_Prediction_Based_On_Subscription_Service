import streamlit as st
import pandas as pd
import numpy as np
import base64
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Streamlit app configuration
st.set_page_config(
    page_title="Churn Prediction",
    page_icon='<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="white" class="bi bi-bag-check-fill" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M10.5 3.5a2.5 2.5 0 0 0-5 0V4h5zm1 0V4H15v10a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V4h3.5v-.5a3.5 3.5 0 1 1 7 0m-.646 5.354a.5.5 0 0 0-.708-.708L7.5 10.793 6.354 9.646a.5.5 0 1 0-.708.708l1.5 1.5a.5.5 0 0 0 .708 0z"/></svg>',
    layout="centered",
    initial_sidebar_state="auto",
)
data = pd.read_csv('train.csv')
# Function to load and preprocess data
@st.cache_resource
def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('train.csv')
    
    # Drop the specified columns
    columns_to_drop = ["PaymentMethod", "PaperlessBilling", "GenrePreference", "Gender", "ContentType",
                       "SupportTicketsPerMonth", "ParentalControl", "DeviceRegistered", "CustomerID"]
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Encode categorical features
    df['MultiDeviceAccess'] = df['MultiDeviceAccess'].map({'No': 0, 'Yes': 1})
    df['SubtitlesEnabled'] = df['SubtitlesEnabled'].map({'No': 0, 'Yes': 1})
    df['SubscriptionType'] = df['SubscriptionType'].map({'Basic': 0, 'Standard': 2, 'Premium': 1})
    
    # Standardize numerical features
    ss = StandardScaler()
    df[['AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
        'AverageViewingDuration', 'ContentDownloadsPerMonth', 'UserRating', 'WatchlistSize']] = ss.fit_transform(
        df[['AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
            'AverageViewingDuration', 'ContentDownloadsPerMonth', 'UserRating', 'WatchlistSize']])
    
    return df

# Function to split data and perform SMOTE
@st.cache_resource
def split_and_resample_data(df):
    X = df.drop(columns="Churn", axis=1)
    y = df["Churn"]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(Xtrain, ytrain)
    
    return X_train_resampled, y_train_resampled

# Function to train the LGBM model
@st.cache_resource
def train_lgbm_model(X_train_resampled, y_train_resampled):
    lgbm_classifier = LGBMClassifier(max_depth=-1, n_estimators=300, subsample=0.6)
    lgbm_classifier.fit(X_train_resampled, y_train_resampled)
    return lgbm_classifier

# Function to set background image
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        opacity: 0.8;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Load and preprocess data
df = load_and_preprocess_data()

# Split and resample data
X_train_resampled, y_train_resampled = split_and_resample_data(df)

# Train the LGBM model
lgbm_classifier = train_lgbm_model(X_train_resampled, y_train_resampled)

# Set background
set_background('Background.jpg')

# Styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: green;
        text-align: center;
        margin-bottom: 2rem;
    }
    .title1 {
        color: red;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title('Customer Churn Prediction For Subscription')

# Input form
with st.form(key='input_form'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        acc_age = st.number_input('Account Age', format="%.15f",value=None)
    with col2:
        monthly_charges = st.number_input('Monthly Charges', format="%.15f",value=None)
    with col3:
        total_charges = st.number_input('Total Charges', format="%.15f",value=None)
    with col4:
        viewing_hours_per_week = st.number_input('Viewing Hours Per Week', format="%.15f",value=None)
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        average_viewing_duration = st.number_input('Average Viewing Duration', format="%.15f",value=None)
    with col6:
        content_downloads_per_month = st.number_input('Content Downloads/mon', format="%.15f",value=None)
    with col7:
        user_rating = st.number_input('User Rating', format="%.15f",value=None)
    with col8:
        watchlist_size = st.number_input('Watchlist Size', format="%.15f",value=None)
    
    col9, col10, col11 = st.columns(3)
    with col9:
        subscription_type = st.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
    with col10:
        multi_device_access = st.selectbox('MultiDevice Access', ['Yes', 'No'])
    with col11:
        subtitles_enabled = st.selectbox('Subtitles Enabled', ['Yes', 'No'])
    col12, col13,col14,col15,col16 = st.columns(5)
    with col14:
        submit_button = st.form_submit_button(label='Predict the Churn')

# Standardize the input data
if submit_button:
    standard = pd.DataFrame({
        'AccountAge': [acc_age],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'ViewingHoursPerWeek': [viewing_hours_per_week],
        'AverageViewingDuration': [average_viewing_duration],
        'ContentDownloadsPerMonth': [content_downloads_per_month],
        'UserRating': [user_rating],
        'WatchlistSize': [watchlist_size]
    })
    
    # Standardizing the input features
    ss=StandardScaler()
    X = data[['AccountAge','MonthlyCharges','TotalCharges','ViewingHoursPerWeek','AverageViewingDuration','ContentDownloadsPerMonth','UserRating','WatchlistSize','SubscriptionType','MultiDeviceAccess','SubtitlesEnabled']]
    X = pd.concat([X, standard], ignore_index=True)
    X[['AccountAge','MonthlyCharges','TotalCharges','ViewingHoursPerWeek','AverageViewingDuration','ContentDownloadsPerMonth','UserRating','WatchlistSize']] = ss.fit_transform(X[['AccountAge','MonthlyCharges','TotalCharges','ViewingHoursPerWeek','AverageViewingDuration','ContentDownloadsPerMonth','UserRating','WatchlistSize']])
    value = X.iloc[-1, :].values.tolist()

    
    # Mapping categorical features
    subscription_mapping = {'Basic': 0, 'Standard': 2, 'Premium': 1}
    subscriptiontype = subscription_mapping.get(subscription_type)
    
    multi_device_mapping = {'Yes': 1, 'No': 0}
    multidevice = multi_device_mapping.get(multi_device_access)
    
    subtitles_mapping = {'Yes': 1, 'No': 0}
    subtitles = subtitles_mapping.get(subtitles_enabled)
    
    input_data = pd.DataFrame({
        'AccountAge': [value[0]],
        'MonthlyCharges': [value[1]],
        'TotalCharges': [value[2]],
        'SubscriptionType': [subscriptiontype],
        'MultiDeviceAccess': [multidevice],
        'ViewingHoursPerWeek': [value[3]],
        'AverageViewingDuration': [value[4]],
        'ContentDownloadsPerMonth': [value[5]],
        'UserRating': [value[6]],
        'WatchlistSize': [value[7]],
        'SubtitlesEnabled': [subtitles]
    })
    
    # Predicting with the model
    prediction = lgbm_classifier.predict(input_data)
    if prediction[0] == 0:
        st.markdown(f"<h2 class='title'>Yes!! This user will subscribe 😊</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 class='title1'>Sorry!! No more subscription 😔</h2>", unsafe_allow_html=True)
