import pickle

from flask import Flask, jsonify, request

with open("./models/2022-01.bin", "rb") as f_in:
    model = pickle.load(f_in)

# trip = {
#     "PULocationID": "43",
#     "DOLocationID": "238",
#     "trip_distance": 1.16,
# }
# prediction = model.predict(trip)
# print(prediction[0])


def prepare_features(ride):
    features = dict()
    features["PULocationID"] = str(ride["PULocationID"])
    features["DOLocationID"] = str(ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask("duration-predication")


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.json
    features = prepare_features(ride)
    pred = predict(features)
    result = {
        "prediction": {
            "duration": pred,
        }
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
