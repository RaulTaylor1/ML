from flask import Flask, request, render_template, jsonify
import pickle
import os
import numpy as np

os.chdir(os.path.dirname(__file__))
print(os.getcwd())

model = pickle.load(open('modelo_CB_entrenado2.pkl', 'rb'))

print(model)

app = Flask(__name__)

@app.route('/')
def index():
    # Renderizar la página principal con la foto de fondo y el botón 'calcular precio'
    return render_template('home.html')

@app.route('/predict_input')
def predict_input():
    # Renderizar la página para ingresar los datos de predicción
    return render_template('predict_input.html')

# Ruta para la segunda página
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    year = int(request.form['year'])
    kilometers_driven = float(request.form['kilometers_driven'])
    fuel_type = int(request.form['fuel_type'])
    transmission = int(request.form['transmission'])
    owner_type = int(request.form['owner_type'])
    mileage = float(request.form['mileage'])
    engine = float(request.form['engine'])
    power = float(request.form['power'])
    seats = int(request.form['seats'])
    
    # Preprocesar los datos
    input_data = np.array([[year, kilometers_driven, fuel_type, transmission, owner_type, mileage, engine, power, seats]])
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    
    # Devolver el resultado de la predicción
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)