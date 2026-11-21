import json
import sys
from pathlib import Path

from flight_price_prediction.pipeline import predict_fares


def main() -> None:
    record = json.load(sys.stdin)
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "outputs" / "models" / "flight_price_model.joblib"
    prediction = predict_fares(model_path, [record])[0]
    print(json.dumps({"prediction": prediction}))


if __name__ == "__main__":
    main()
