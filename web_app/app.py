from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs
import json
import os
import sys

import joblib
import pandas as pd


# Project root folder. Used to load files from dataset/ and models/.
BASE_DIR = Path(__file__).resolve().parents[1]

# Folder that contains the web app files such as index.html and styles.css.
APP_DIR = Path(__file__).resolve().parent

# Load the final tuned model that will be used for CKD prediction.
model = joblib.load(BASE_DIR / "models" / "xGBoostModelTuned.pkl")

# Load the exact feature column names used during training.
# The prediction input must match these columns and their order.
columns = list(pd.read_csv(BASE_DIR / "dataset" / "XTrain.csv", nrows=0).columns)

# Form fields that should be converted from text input into numeric values.
numeric_fields = [
    "age", "bmi", "systolicBp", "diastolicBp", "serumCreatinine",
    "egfr", "bloodUrea", "hemoglobin", "urineProtein",
    "urineSpecificGravity", "bloodGlucose"
]


def page(result_html=""):
    # Load the HTML page and replace the result placeholder if a result is given.
    html = (APP_DIR / "index.html").read_text()
    return html.replace("{{ result }}", result_html)


def build_input(form):
    # Start with every model feature set to 0.
    # This ensures one-hot encoded columns exist even if they are not selected.
    row = {column: 0 for column in columns}

    # Convert numeric form inputs from strings into floats for the model.
    for field in numeric_fields:
        row[field] = float(form[field][0])

    # Recreate the same one-hot encoded categorical columns used in training.
    row["gender_Male"] = int(form["gender"][0] == "Male")
    row["diabetesMellitus_Yes"] = int(form["diabetesMellitus"][0] == "Yes")
    row["hypertension_Yes"] = int(form["hypertension"][0] == "Yes")
    row["smokingStatus_Former"] = int(form["smokingStatus"][0] == "Former")
    row["smokingStatus_Never"] = int(form["smokingStatus"][0] == "Never")
    row["familyHistoryCkd_Yes"] = int(form["familyHistoryCkd"][0] == "Yes")
    row["anemia_Yes"] = int(form["anemia"][0] == "Yes")
    row["pedalEdema_Yes"] = int(form["pedalEdema"][0] == "Yes")

    # Return a one-row DataFrame with the exact same column order as training.
    return pd.DataFrame([row], columns=columns)


def predict(form):
    # Convert the submitted form into model-ready feature data.
    patient_data = build_input(form)

    # Get the probability for class 1, which represents CKD.
    ckd_probability = model.predict_proba(patient_data)[0][1]

    # Use 0.5 as the decision threshold for the final class label.
    label = "CKD" if ckd_probability >= 0.5 else "No CKD"

    # Return the prediction result in a frontend-friendly format.
    return {
        "label": label,
        "probability": f"{ckd_probability * 100:.1f}%"
    }


class App(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve the main page.
        if self.path == "/":
            self.send(page())

        # Serve the stylesheet used by the page.
        elif self.path == "/styles.css":
            self.send((APP_DIR / "styles.css").read_text(), "text/css")

        # Return 404 for any unknown GET route.
        else:
            self.send_error(404)

    def do_POST(self):
        # The app only accepts form submissions at /predict.
        if self.path != "/predict":
            self.send_error(404)
            return

        # Read and parse the submitted form data.
        length = int(self.headers["Content-Length"])
        form = parse_qs(self.rfile.read(length).decode())

        # Run the model prediction and return the result as JSON.
        self.send_json(predict(form))

    def send(self, content, content_type="text/html"):
        # Send a successful text response such as HTML or CSS.
        body = content.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data):
        # Convert a Python dictionary into a JSON HTTP response.
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(ThreadingHTTPServer):
    # Allow the same port to be reused soon after restarting the server.
    allow_reuse_address = True


if __name__ == "__main__":
    # Use PORT from the environment if provided; otherwise default to 8000.
    port = int(os.environ.get("PORT", "8000"))
    try:
        print(f"Open http://127.0.0.1:{port}")

        # Start the local web server and keep it running until stopped.
        Server(("127.0.0.1", port), App).serve_forever()
    except OSError as error:
        # Show a helpful message when the selected port is already in use.
        if error.errno == 48:
            print(f"Port {port} is already in use.")
            print("Stop the other server with Ctrl+C, or run this app on another port:")
            print("PORT=8001 python3 app.py")
            sys.exit(1)

        # Re-raise any other server startup error.
        raise
