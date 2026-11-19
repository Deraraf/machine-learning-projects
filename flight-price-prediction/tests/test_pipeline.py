import unittest

import numpy as np
import pandas as pd

from flight_price_prediction.pipeline import clean_flight_data, make_features


def sample_flight(price: float = 5000) -> dict[str, object]:
    return {
        "airline": "Example Air",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "zero",
        "arrival_time": "Afternoon",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": 2.5,
        "days_left": 10,
        "flight": "EX-123",
        "price": price,
    }


class FlightDataPreparationTests(unittest.TestCase):
    def test_cleaning_drops_duplicate_rows_and_invalid_targets(self) -> None:
        first = sample_flight()
        duplicate = sample_flight()
        invalid_target = sample_flight()
        invalid_target["price"] = np.nan
        frame = pd.DataFrame(
            [first, duplicate, invalid_target],
            columns=["Unnamed: 0", *first.keys()],
        )
        frame["Unnamed: 0"] = range(len(frame))

        cleaned, report = clean_flight_data(frame)

        self.assertEqual(len(cleaned), 1)
        self.assertNotIn("Unnamed: 0", cleaned.columns)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["invalid_targets_removed"], 1)

    def test_features_add_route_and_exclude_fare_and_flight_identifier(self) -> None:
        frame = pd.DataFrame([sample_flight()])

        features = make_features(frame)

        self.assertEqual(features.loc[0, "route"], "Delhi -> Mumbai")
        self.assertNotIn("price", features.columns)
        self.assertNotIn("flight", features.columns)
        self.assertEqual(features.loc[0, "duration"], 2.5)

    def test_features_reject_missing_required_input(self) -> None:
        frame = pd.DataFrame([{"airline": "Example Air"}])

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            make_features(frame)


if __name__ == "__main__":
    unittest.main()
