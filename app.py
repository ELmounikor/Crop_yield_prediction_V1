from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pickle


preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
dnn = load_model('dnn.h5')
app = Flask(__name__)


def predictor(area:str, item:str, year:int, average_rain_fall_mm_per_year:float, pesticides_tonnes:float, avg_temp:float):
    features = np.array([[area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])
    predicted_value = dnn.predict(preprocessor.transform(features).toarray())
    return predicted_value[0][0]


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        area = request.form['area']
        item = request.form['item']
        year = request.form['year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        predicted_value = predictor(area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp)
        return render_template('index.html', predicted_value=predicted_value)

if __name__ == "__main__":
    app.run(debug=True)
