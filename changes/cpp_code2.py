import streamlit as st
import pandas as pd
import pickle


# --- Page Config ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_nlp = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_qp1q7mct.json")
# col1, col2 = st.columns([2, 3])
# with col1:
#     st_lottie(lottie_nlp, height=160, speed=1)

# with col2:
st.markdown("""
    <div style="
        background-color: rgba(0, 0, 0, 0.65);
        padding: 15px 20px;
        border-radius: 12px;
        border-left: 5px solid #FF3131;
        color: white;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-top: 10px;
    ">
        <h2 style="margin-bottom: 5px; font-size: 28px;">
            <span style="color:#FF3131;">🚗 Car Price Prediction</span>
        </h2>
        <p style="font-style: italic; font-size: 15px; color: #f0f0f0; margin-top: 0;">
            – Find the accurate resale price of your car using <b style="color: #FF3131;">machine learning</b>!
        </p>
    </div>
""", unsafe_allow_html=True)


#st.markdown("<h4 style='text-align: center;'>Find the accurate resale price of your car using machine learning!</h4>", unsafe_allow_html=True)

# --- Load Model & Data ---
with open('LinearRegressionModel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dataframe_car.pkl', 'rb') as f:
    df = pickle.load(f)



import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/png;base64,{encoded_string}");
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
             background-attachment: fixed;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Call this before your Streamlit content
add_bg_from_local("bg_car2.jpeg")

st.sidebar.image("car-Logos.png", use_column_width=True)

# --- Sidebar Inputs ---
st.sidebar.header("🛠️ Car Information")

selected_company = st.sidebar.selectbox(
    "Select Car Company",
    options=sorted(df['company'].unique()),
    placeholder="Select a company"
)


filtered_models = df[df['company'] == selected_company]['name'].unique()

selected_model = st.sidebar.selectbox(
    "Select Car Model",
    options=sorted(filtered_models),
    placeholder="Select a model"
)

manufacture_year = st.sidebar.select_slider(
    "Manufacturing Year",
    options=list(range(1960, 2026)),
    value=2015
)

kms_driven = st.sidebar.text_input(
    "Kilometers Driven",
    placeholder="e.g. 45200"
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    options=sorted(df['fuel_type'].unique())
)

# --- Prediction Trigger ---
if st.sidebar.button("🚀 Predict Price"):
    if not kms_driven.strip().isdigit():
        st.error("❌ Please enter a valid numeric value for kilometers driven.")
    else:
        kms_driven = int(kms_driven)
        input_df = pd.DataFrame({
            'name': [selected_model],
            'company': [selected_company],
            'year': [manufacture_year],
            'kms_driven': [kms_driven],
            'fuel_type': [fuel_type]
        })

        # Predict
        predicted_price = model.predict(input_df)[0]

        st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.15);
                border-radius: 16px;
                padding: 20px;
                text-align: center;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.3);
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                font-size: 26px;
                font-weight: 600;
                color: white;
                margin-top: 30px;
            ">
                💰 Estimated Price: ₹ {predicted_price:,.2f}
            </div>
        """, unsafe_allow_html=True)


# --- Footer ---
st.markdown("---")

st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f5f5f5;
            color: #444;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
    </style>
    <div class="footer">
        Made with ❤️ by <b>Bhumika Sharma</b> | Streamlit + Machine Learning | bhumikasharma1808@gmail.com
    </div>
""", unsafe_allow_html=True)

