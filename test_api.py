import unittest
from fastapi.testclient import TestClient
from main import app

class TestPredictiveAnalyticsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_list_datasets(self):
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("datasets", data)
        self.assertIn("sample_housing.csv", data["datasets"])
        self.assertIn("sample_sales.csv", data["datasets"])

    def test_dataset_details_housing(self):
        response = self.client.get("/api/datasets/sample_housing.csv")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "sample_housing.csv")
        self.assertIn("shape", data)
        self.assertIn("columns", data)
        self.assertIn("preview", data)
        
        # Verify columns exist
        col_names = [col["name"] for col in data["columns"]]
        self.assertIn("PriceUSD", col_names)
        self.assertIn("SquareFeet", col_names)

    def test_train_regression_linear(self):
        payload = {
            "dataset_name": "sample_housing.csv",
            "target_column": "PriceUSD",
            "feature_columns": ["SquareFeet", "Bedrooms", "Bathrooms", "AgeYears"],
            "imputation_strategies": {},
            "scale_features": True,
            "scaler_type": "standard",
            "model_type": "linear",
            "test_size": 0.2,
            "hyperparameters": {}
        }
        response = self.client.post("/api/train/regression", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
        self.assertIn("predictions", data)
        self.assertIn("feature_importances", data)
        self.assertIn("residuals", data)
        
        # Check metrics format
        self.assertIn("r2", data["metrics"]["test"])
        self.assertIn("mae", data["metrics"]["test"])

    def test_train_timeseries_arima(self):
        payload = {
            "dataset_name": "sample_sales.csv",
            "date_column": "Month",
            "target_column": "Sales",
            "model_type": "arima",
            "test_periods": 12,
            "forecast_horizon": 12,
            "hyperparameters": {"p": 1, "d": 1, "q": 1}
        }
        response = self.client.post("/api/train/timeseries", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("metrics", data)
        self.assertIn("historical", data)
        self.assertIn("fitted", data)
        self.assertIn("predictions", data)
        self.assertIn("forecast", data)
        
        # Check MAPE is calculated
        self.assertIn("mape", data["metrics"])

if __name__ == "__main__":
    unittest.main()
