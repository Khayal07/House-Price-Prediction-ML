import gradio as gr
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

model = joblib.load('src/model.pkl')


df = pd.read_csv('data/data.csv')
le = LabelEncoder()
le.fit(df['city'])
unique_cities = sorted(df['city'].unique().tolist())


def predict_price(bedrooms, bathrooms, sqft_living, sqft_above, city_name):
    city_encoded = le.transform([city_name])[0]
    
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': 8000,
        'floors': 1.5,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'sqft_above': sqft_above,
        'sqft_basement': 0,
        'yr_built': 1970,
        'yr_renovated': 0,
        'city': city_encoded,
        'house_age': 2026 - 1970,   # new column
        'is_renovated': 0,          # new column
        'total_area': sqft_living + 8000 # new column
    }])
    
    prediction_log = model.predict(input_data)[0]
    
    prediction = np.expm1(prediction_log)
    
    return f"${prediction:,.2f}"

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🏠 USA House Price Predictor")
    gr.Markdown("Enter the details of the house below to get an estimated price based on our Machine Learning model.")
    
    with gr.Row():
        with gr.Column():
            bedrooms = gr.Slider(minimum=1, maximum=10, step=1, label="Bedrooms")
            bathrooms = gr.Slider(minimum=1, maximum=8, step=0.5, label="Bathrooms")
            city = gr.Dropdown(choices=unique_cities, label="City", value="Seattle")
        
        with gr.Column():
            sqft_living = gr.Number(label="Living Area (sqft)", value=1500)
            sqft_above = gr.Number(label="Above Ground Area (sqft)", value=1500)
            
    predict_btn = gr.Button("Predict Price", variant="primary")
    output = gr.Textbox(label="Estimated Price", text_align="center", scale=2)
    
    predict_btn.click(
        fn=predict_price,
        inputs=[bedrooms, bathrooms, sqft_living, sqft_above, city],
        outputs=output
    )

if __name__ == "__main__":
    interface.launch(share=False)