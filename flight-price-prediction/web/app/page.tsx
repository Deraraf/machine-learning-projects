"use client";

import { SubmitEvent, useEffect, useState } from "react";

type Status = {
  selectedModel: string;
  testMetrics: { mae: number; rmse: number; r2: number };
  cleaning: { clean_rows: number };
};

const options = {
  airline: ["Vistara", "Air_India", "Indigo", "GO_FIRST", "AirAsia", "SpiceJet"],
  source_city: ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"],
  departure_time: ["Morning", "Early_Morning", "Afternoon", "Evening", "Night", "Late_Night"],
  stops: ["zero", "one", "two_or_more"],
  arrival_time: ["Morning", "Early_Morning", "Afternoon", "Evening", "Night", "Late_Night"],
  destination_city: ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"],
  class: ["Economy", "Business"],
};

export default function Home() {
  const [status, setStatus] = useState<Status | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    airline: "Vistara",
    source_city: "Delhi",
    departure_time: "Morning",
    stops: "zero",
    arrival_time: "Evening",
    destination_city: "Mumbai",
    class: "Economy",
    duration: 2,
    days_left: 30,
  });

  useEffect(() => {
    fetch("/api/status")
      .then((response) => response.json())
      .then((data) => {
        if (data.error) setError(data.error);
        else setStatus(data);
      })
      .catch(() => setError("Could not read project status."));
  }, []);

  async function predict(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed");
      setPrediction(data.prediction);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function update(field: keyof typeof form, value: string | number) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function selectField(field: keyof typeof options, label: string) {
    return (
      <label key={field}>
        <span>{label}</span>
        <select
          value={form[field]}
          onChange={(event) => update(field, event.target.value)}
        >
          {options[field].map((option) => (
            <option key={option} value={option}>
              {option.replaceAll("_", " ")}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">Machine Learning Project</p>
          <h1>Flight Fare Predictor</h1>
          <p className="subtitle">
            Enter itinerary details to estimate the listed ticket price using the
            trained regression model.
          </p>
        </div>
        {status && (
          <aside className="model-card">
            <span>Selected model</span>
            <strong>{status.selectedModel}</strong>
            <small>{status.cleaning.clean_rows.toLocaleString()} clean records</small>
          </aside>
        )}
      </section>

      <section className="workspace">
        <form onSubmit={predict}>
          <div className="form-grid">
            {selectField("airline", "Airline")}
            {selectField("class", "Travel class")}
            {selectField("source_city", "Source city")}
            {selectField("destination_city", "Destination city")}
            {selectField("departure_time", "Departure time")}
            {selectField("arrival_time", "Arrival time")}
            {selectField("stops", "Stops")}

            <label>
              <span>Duration (hours)</span>
              <input
                type="number"
                min="0.1"
                max="50"
                step="0.1"
                value={form.duration}
                onChange={(event) => update("duration", Number(event.target.value))}
                required
              />
            </label>

            <label>
              <span>Days left</span>
              <input
                type="number"
                min="1"
                max="365"
                step="1"
                value={form.days_left}
                onChange={(event) => update("days_left", Number(event.target.value))}
                required
              />
            </label>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Calculating..." : "Predict flight fare"}
          </button>
        </form>

        <aside className="results">
          <p className="result-label">Estimated fare</p>
          {prediction !== null ? (
            <div className="prediction">
              <span>₹</span>
              {prediction.toLocaleString("en-IN", {
                maximumFractionDigits: 0,
              })}
            </div>
          ) : (
            <div className="placeholder">Waiting for input</div>
          )}
          {error && <p className="error">{error}</p>}

          {status && (
            <div className="metrics">
              <h2>Test performance</h2>
              <div>
                <span>R²</span>
                <strong>{status.testMetrics.r2.toFixed(3)}</strong>
              </div>
              <div>
                <span>MAE</span>
                <strong>₹{status.testMetrics.mae.toFixed(0)}</strong>
              </div>
              <div>
                <span>RMSE</span>
                <strong>₹{status.testMetrics.rmse.toFixed(0)}</strong>
              </div>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
