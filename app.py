from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd # Use 'pd' for convention
import os

app = Flask(__name__)

script_dir = os.path.dirname(__file__)
model_pipeline_path = os.path.join(script_dir, "model.pkl")

try:
    with open(model_pipeline_path, 'rb') as model_file:
        full_pipeline = pickle.load(model_file)
    print("Full model pipeline loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model pipeline: {e}. Make sure 'model.pkl' is in the same directory as 'app.py'.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during model pipeline loading: {e}")
    exit()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            # --- START DEBUGGING PRINTS ---
            print(f"\n--- DEBUG INFO ---")
            print(f"1. Received raw form data from client: {form_data}") # Shows what HTML sent
            # --- END DEBUGGING PRINTS ---

            original_feature_names = [
                'holiday', 'temp', 'rain', 'snow', 'weather', 'year',
                'month', 'day', 'hours', 'minutes', 'seconds'
            ]

            input_dict = {}
            for name in original_feature_names:
                value = form_data.get(name)

                if value is None or value == '':
                    raise ValueError(f"Missing or empty input for '{name}'. Please fill all fields.")

                if name in ['holiday', 'weather']:
                    input_dict[name] = [str(value)]
                elif name in ['temp', 'rain', 'snow']:
                    input_dict[name] = [float(value)]
                elif name in ['year', 'month', 'day', 'hours', 'minutes', 'seconds']:
                    input_dict[name] = [int(value)]

            input_df = pd.DataFrame(input_dict, columns=original_feature_names)
            # --- START CRITICAL DEBUGGING PRINTS ---
            print(f"2. DataFrame prepared for prediction (check column order and types):\n{input_df}")
            print(f"3. DataFrame dtypes:\n{input_df.dtypes}")
            print(f"--- END DEBUG INFO ---\n")
            # --- END CRITICAL DEBUGGING PRINTS ---

            prediction = full_pipeline.predict(input_df)
            print(f"Raw prediction: {prediction}")

            text = "Estimated Traffic Volume is: "
            return render_template("output.html", prediction_text=text + str(round(prediction[0], 2)))

        except ValueError as ve:
            print(f"ValueError during prediction: {ve}")
            return render_template("index.html", error_message=f"Input Error: {ve}. Please ensure all fields are correctly filled with valid data.")
        except KeyError as ke:
            print(f"KeyError during prediction: {ke}")
            return render_template("index.html", error_message=f"Form Error: Missing input for field '{ke}'. Please check your HTML form names and ensure all fields are submitted.")
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            # This is the last resort error, if this hits, it means the pipeline.predict() itself failed
            # Print the dataframe that caused the error for more context
            # print(f"Error occurred with DataFrame:\n{input_df}") # Uncomment if you want to see the df even on general exception
            return render_template("index.html", error_message=f"An unexpected error occurred: {e}. Check server logs for details.")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)