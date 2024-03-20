import azure.functions as func
import logging

from utils.helper_functions import initialize_model, predict

# ------------------------------------------------------------------------------
app = func.FunctionApp()
model = initialize_model()


# ------------------------------------------------------------------------------
@app.route(route="health")
def health(req: func.HttpRequest) -> dict:
    logging.info("Health endpoint triggered.")
    return {"status": "healthy"}


# ------------------------------------------------------------------------------
@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest):
    logging.info("Predict endpoint triggered.")
    prediction = predict(model, req.get_body())
    logging.info(f"Prediction made.")
    return prediction[1]
