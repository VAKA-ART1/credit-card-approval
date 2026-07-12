from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Dashboard Data
total_predictions = 0
approved_count = 0
rejected_count = 0

last_name = ""
last_result = ""
last_confidence = ""
last_risk = ""
prediction_history = []

# Load model and encoders
model = joblib.load("models/credit_card_model.pkl")
encoders = joblib.load("models/encoders.pkl")


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION PAGE ----------------
@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    global total_predictions
    global approved_count
    global rejected_count
    global last_name
    global last_result
    global last_confidence
    global last_risk
    global prediction_history

    # Personal Information
    name = request.form["name"]
    email = request.form["email"]
    phone = request.form["phone"]

    gender_text = request.form["gender"]
    own_car_text = request.form["own_car"]
    own_realty_text = request.form["own_realty"]

    age = int(request.form["age"] or 0)
    children = int(request.form["children"] or 0)

    # Employment
    income = float(request.form["income"] or 0)
    income_type_text = request.form["income_type"]
    occupation_text = request.form["occupation"]
    employment_days = int(request.form["employment_days"] or 0)

    # Education
    education_text = request.form["education"]
    family_status_text = request.form["family_status"]

    # Housing
    housing_text = request.form["housing"]
    family_members = float(request.form["family_members"] or 1)

    # Phone / Email Flags
    work_phone = int(request.form["work_phone"])
    phone_flag = int(request.form["phone_flag"])
    email_flag = int(request.form["email_flag"])

    # Encode categorical values
    gender = encoders["CODE_GENDER"].transform([gender_text])[0]
    own_car = encoders["FLAG_OWN_CAR"].transform([own_car_text])[0]
    own_realty = encoders["FLAG_OWN_REALTY"].transform([own_realty_text])[0]
    income_type = encoders["NAME_INCOME_TYPE"].transform([income_type_text])[0]
    education = encoders["NAME_EDUCATION_TYPE"].transform([education_text])[0]
    family_status = encoders["NAME_FAMILY_STATUS"].transform([family_status_text])[0]
    housing = encoders["NAME_HOUSING_TYPE"].transform([housing_text])[0]
    occupation = encoders["OCCUPATION_TYPE"].transform([occupation_text])[0]

    # Create feature array
    features = np.array([[
        gender,
        own_car,
        own_realty,
        children,
        income,
        income_type,
        education,
        family_status,
        housing,
        -age * 365,
        employment_days,
        1,
        work_phone,
        phone_flag,
        email_flag,
        occupation,
        family_members
    ]])

    # ---------------- Prediction ----------------

    # Bank Rule
    if (
        income < 50000
        and employment_days == 0
        and income_type_text == "Student"
        and own_car_text == "N"
        and own_realty_text == "N"
    ):
        result = "Rejected ❌"
        risk = "High Risk"
        confidence = "95.00"

        # Update Dashboard
        total_predictions += 1
        if result == "Approved ✅":
            approved_count += 1
        else:
            rejected_count += 1
            
        last_name = name
        last_result = result
        last_confidence = confidence + "%"
        last_risk = risk

    else:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = str(round(max(probability) * 100, 2))

        if prediction == 0:
            result = "Approved ✅"
            risk = "Low Risk"
        else:
            result = "Rejected ❌"
            risk = "High Risk"

        # Update Dashboard
        total_predictions += 1

        if result == "Approved ✅":
            approved_count += 1
        else:
            rejected_count += 1

        last_name = name
        last_result = result
        last_confidence = confidence + "%"
        last_risk = risk

    prediction_history.append({
        "name": name,
        "result": result,
        "confidence": confidence + "%",
        "risk": risk
      })
    
    return render_template(
        "result.html",
        result=result,
        confidence=confidence + "%",
        risk=risk,
        name=name,
        email=email,
        phone=phone,
        gender=gender_text,
        age=age,
        children=children,
        income=income,
        income_type=income_type_text,
        occupation=occupation_text,
        employment_days=employment_days,
        education=education_text,
        family_status=family_status_text,
        housing=housing_text,
        family_members=family_members,
        months_balance=request.form.get("months_balance", ""),
        payment_status=request.form.get("payment_status", ""),
        overdue_status=request.form.get("overdue_status", "")
    )

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        total=total_predictions,
        approved=approved_count,
        rejected=rejected_count,
        history=prediction_history
    )
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
