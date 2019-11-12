import joblib
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
     json_ = [request.json]
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = lr.predict(query).tolist()[0]
     return jsonify({'prediction': prediction})
     #return str(prediction)	
if __name__ == '__main__':
    port = 5005 # If you don't provide any port then the port will be set to 5005
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    #app.run(port=port, debug=True)
    app.run(debug=True,host='0.0.0.0',port=port)
