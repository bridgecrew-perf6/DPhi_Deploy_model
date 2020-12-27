#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

#app name
app = Flask(__name__)
model = pickle.load(open('Loan_mdl.pkl', 'rb'))

#home page
@app.route('/')
def home():
    return render_template('index.html')

#predict the result and return it
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    labels = ['Rejected', 'Approved']

    features = [int(x) for x in request.form.values()]
   
    values = [np.array(features)]
    
    prediction = model.predict(values)

    result = labels[prediction[0]]

    return render_template('index.html', output='{}'.format(result))


if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)