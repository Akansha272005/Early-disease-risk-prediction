from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
CORS(app)


DATA_PATH = "health_data1.csv"
data = pd.read_csv(DATA_PATH)




data.columns = data.columns.str.lower()


risk_map = {'low': 0, 'medium': 1, 'high': 2}
if 'health_risk' in data.columns:
    data['health_risk'] = data['health_risk'].map(risk_map)
data['health_risk'] = data['health_risk'].fillna(0)


binary_map = {'yes': 1, 'no': 0}
binary_columns = ['exercise','smoking','alcohol','married']
for col in binary_columns:
    if col in data.columns:
        data[col] = data[col].map(binary_map).fillna(0)


feature_cols = ['age','bmi','exercise','sleep','sugar_intake','smoking','alcohol','married']
for col in feature_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)


X = data[feature_cols]
y = data['health_risk']


model = LogisticRegression(max_iter=1000)
model.fit(X, y)

@app.route('/')
def home():
    return "EARLY DISEASE RISK PREDICTION Backend Running"

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    try:
        input_data = [[
            float(d['age']),
            float(d['bmi']),
            int(d['exercise']),
            float(d['sleep']),
            float(d['sugar_intake']),
            int(d['smoking']),
            int(d['alcohol']),
            int(d['married'])
        ]]
        pred = model.predict(input_data)[0]

        
        proba = model.predict_proba(input_data)
        prob = proba[0][pred] if pred < len(proba[0]) else max(proba[0])

        risk_label = {0:"Low Risk", 1:"Medium Risk", 2:"High Risk"}

        
        if not os.path.exists("static"):
            os.makedirs("static")
        plt.figure()
        plt.bar(["Risk Probability"], [prob*100], color='red')
        plt.ylim(0,100)
        plt.ylabel("Probability (%)")
        plt.title("Disease Risk Prediction")
        plt.savefig("static/graph.png")
        plt.close()

        return jsonify({
            "risk": risk_label.get(pred,"Unknown"),
            "probability": round(prob*100,2),
            "graph_url": "/static/graph.png"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
