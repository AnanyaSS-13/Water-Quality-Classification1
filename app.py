from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("water_potability_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get input values from the form
    ph = float(request.form["ph"])
    hardness = float(request.form["hardness"])
    solids = float(request.form["solids"])
    chloramines = float(request.form["chloramines"])
    sulfate = float(request.form["sulfate"])
    conductivity = float(request.form["conductivity"])
    organic_carbon = float(request.form["organic_carbon"])
    trihalomethanes = float(request.form["trihalomethanes"])
    turbidity = float(request.form["turbidity"])

    # Create input data in the same order used during model training
    new_water = [[
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity
    ]]

    # Make prediction
    prediction = model.predict(new_water)

    # Display result
    if prediction[0] == 1:
        result = "POTABLE - Water is suitable based on the model"
    else:
        result = "NOT POTABLE - Water is not suitable based on the model"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)