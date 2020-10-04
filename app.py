import numpy as np
from flask import Flask , request , jsonify , render_template
from joblib import dump, load 



app = Flask(__name__)
model = load('F:\Machine Learning\Diabetes Prediction\DIABETESPRED.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict' , methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    return render_template('index.html' , prediction_text = prediction[0])

@app.route('/1') 
def home1():
    name = "Rohit"
    return render_template('Hello.php' , name = name)

if __name__ == "__main__":
    app.run(debug = True)
