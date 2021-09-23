import flask
from flask import request, Flask, render_template
from joblib import dump, load
app = Flask(__name__)
import numpy as np

@app.route('/')
def default():
    return '<h1>API is working</h1>'

@app.route('/predict')
def predict():
    model = load('marriage_age_predict_model.ml') 
    age_predict = model.predict([[request.args['gender'],
                                  request.args['height'],
                                  request.args['religion'],
                                  request.args['caste'],
                                  request.args['mother_tongue'],
                                  request.args['country']]])
    return str(round(age_predict[0], 2))

@app.route('/home')
def home():
    return render_template('index.html')

# prediction function
def ValuePredictor(to_predict_list):
    model = load('marriage_age_predict_model.ml') 
    to_predict = np.array(to_predict_list).reshape(1, 6)
    result = model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(int, to_predict_list))
		result = ValuePredictor(to_predict_list)	
		
		return render_template("result.html", prediction = round(result, 2))

if __name__ == "__main__":
    app.run(debug=True)

# gender	height	religion	caste	mother_tongue	country