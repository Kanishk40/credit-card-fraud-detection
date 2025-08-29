from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --------- LOAD & PREPROCESS DATA ----------
CSV_PATH = "creditcard.csv"  # make sure this exists in the same folder

# Load dataset
df = pd.read_csv(CSV_PATH)

# We'll use only Time and Amount for a simple model (you can extend later)
data = df[['Time', 'Amount', 'Class']].copy()

# Scale Amount (Time can be kept as-is; optionally scale it too)
scaler = MinMaxScaler()
data['Amount_scaled'] = scaler.fit_transform(data[['Amount']])

# Features and target
X = data[['Time', 'Amount_scaled']].values
y = data['Class'].values

# Train-test split (small holdout, training on most of data)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Use class_weight to help imbalance
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Simple performance print (console)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Model trained. Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")

# Add predictions & anomaly info to df for dashboard sampling
data['pred'] = model.predict(X)
data['score'] = model.predict_proba(X)[:, 1]  # probability of class 1 (fraud)

# --------- ROUTES ----------
@app.route("/")
def home():
    total_outliers = int(data['pred'].sum())
    total_tx = len(data)
    total_inliers = total_tx - total_outliers

    # sample recent / random transactions (for demo)
    sample_tx = df.sample(9, random_state=42).copy()
    # add scaled amount/pred/score columns for view
    sample_tx['Amount_scaled'] = scaler.transform(sample_tx[['Amount']])
    sample_tx['pred'] = model.predict(sample_tx[['Time', 'Amount_scaled']].values)
    sample_tx['score'] = model.predict_proba(sample_tx[['Time', 'Amount_scaled']].values)[:, 1]

    transactions = sample_tx.to_dict(orient="records")

    return render_template("index.html",
                           total_inliers=total_inliers,
                           total_outliers=total_outliers,
                           transactions=transactions)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects form fields:
      - name, email, card_number (masked or last4), expiry (optional)
      - time (seconds), amount (raw USD)
    Returns JSON: result (Legitimate/Fraudulent), score (probability), meta
    """
    try:
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        card_last4 = request.form.get("card_last4", "").strip()
        expiry = request.form.get("expiry", "").strip()

        # read transaction features
        raw_time = float(request.form.get("time", "0"))
        raw_amount = float(request.form.get("amount", "0"))

        # scale amount using the same scaler used in training
        amt_scaled = scaler.transform(np.array(raw_amount).reshape(-1, 1))[0, 0]

        X_input = np.array([[raw_time, amt_scaled]])
        pred = model.predict(X_input)[0]                 # 0 or 1
        prob = model.predict_proba(X_input)[0][1]       # probability of fraud

        result_text = "Fraudulent" if pred == 1 else "Legitimate"

        response = {
            "result": result_text,
            "score": float(prob),
            "name": name,
            "email": email,
            "card_last4": card_last4,
            "expiry": expiry,
            "time": raw_time,
            "amount": raw_amount
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Manual download link for dataset (serves local file)
@app.route("/download-dataset")
def download_dataset():
    return send_file(CSV_PATH, as_attachment=True, download_name="creditcard.csv")

# quick demo link (GET)
@app.route("/quick-predict")
def quick_predict():
    try:
        amt = float(request.args.get("amount", 0))
        t = float(request.args.get("time", 0))
    except ValueError:
        return jsonify({"error": "invalid numeric parameters"}), 400
    amt_scaled = scaler.transform(np.array(amt).reshape(-1, 1))[0, 0]
    pred = model.predict(np.array([[t, amt_scaled]]))[0]
    prob = model.predict_proba(np.array([[t, amt_scaled]]))[0][1]
    return jsonify({"time": t, "amount": amt, "result": "Fraudulent" if pred == 1 else "Legitimate", "score": float(prob)})

if __name__ == "__main__":
    app.run(debug=True)
