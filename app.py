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
            custom_data = CustomData(
                gender=request.form.get("gender"),
                age=request.form.get("age"),
                under_30=request.form.get("under_30"),
                senior_citizen=request.form.get("senior_citizen"),
                married=request.form.get("married"),
                dependents=request.form.get("dependents"),
                num_dependents=request.form.get("num_dependents"),
                country=request.form.get("country"),
                state=request.form.get("state"),
                city=request.form.get("city"),
                zip_code=request.form.get("zip_code"),
                latitude=request.form.get("latitude"),
                longitude=request.form.get("longitude"),
                population=request.form.get("population"),
                quarter=request.form.get("quarter"),
                referred_friend=request.form.get("referred_friend"),
                num_referrals=request.form.get("num_referrals"),
                tenure_months=request.form.get("tenure_months"),
                offer=request.form.get("offer"),
                phone_service=request.form.get("phone_service"),
                multiple_lines=request.form.get("multiple_lines"),
                internet_service=request.form.get("internet_service"),
                internet_type=request.form.get("internet_type"),
                online_security=request.form.get("online_security"),
                online_backup=request.form.get("online_backup"),
                device_protection=request.form.get("device_protection"),
                tech_support=request.form.get("tech_support"),
                streaming_tv=request.form.get("streaming_tv"),
                streaming_movies=request.form.get("streaming_movies"),
                streaming_music=request.form.get("streaming_music"),
                unlimited_data=request.form.get("unlimited_data"),
                avg_long_distance=request.form.get("avg_long_distance"),
                avg_gb_download=request.form.get("avg_gb_download"),
                monthly_charges=request.form.get("monthly_charges"),
                total_charges=request.form.get("total_charges"),
                total_refunds=request.form.get("total_refunds"),
                extra_data_charges=request.form.get("extra_data_charges"),
                long_distance_charges=request.form.get("long_distance_charges"),
                total_revenue=request.form.get("total_revenue"),
                satisfaction_score=request.form.get("satisfaction_score"),
                cltv=request.form.get("cltv"),
                contract=request.form.get("contract"),
                paperless_billing=request.form.get("paperless_billing"),
                payment_method=request.form.get("payment_method")
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
    app.run(host="0.0.0.0", port=10000)
