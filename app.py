from flask import Flask, render_template, request
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline   

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')  


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('result.html')   # Show empty form initially
    else:
        try:
            # ✅ Use CustomData class (instead of manually building dictionary)
            custom_data = CustomData(
                gender=request.form.get("gender"),
                age=request.form.get("age"),
                monthly_charges=request.form.get("monthly_charges"),
                contract=request.form.get("contract")
            )

            # Convert to DataFrame
            data_df = custom_data.get_data_as_dataframe()

            # Run prediction
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(data=data_df)

            # Business explanation
            if result[0] == "Yes":
                prediction_text = (
                    "❌ The customer is likely to CHURN. "
                    "Consider offering discounts, loyalty perks, or retention strategies."
                )
            else:
                prediction_text = (
                    "✅ The customer is NOT likely to churn. "
                    "Keep engaging them with good service to maintain satisfaction."
                )

            return render_template('result.html', prediction=prediction_text)

        except Exception as e:
            return render_template(
                'result.html',
                prediction=f"⚠️ Error occurred: {str(e)}"
            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
