import React, { useState } from "react";
import "./Roadmap.css";

function Roadmap() {
  const [goal, setGoal] = useState("");
  const [timePerDay, setTimePerDay] = useState("1");
  const [learningStyle, setLearningStyle] = useState("visual");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [roadmap, setRoadmap] = useState(null);

  const generateRoadmap = async () => {
    if (!goal.trim()) {
      setError("Please enter a learning goal");
      return;
    }

    setLoading(true);
    setError("");
    setRoadmap(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/learning-roadmap",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            goal,
            time_per_day: timePerDay,
            learning_style: learningStyle,
          }),
        }
      );

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setRoadmap(data);
      }
    } catch (err) {
      setError(
        "Failed to generate roadmap. Make sure Flask backend is running on port 5000."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="roadmap-page">
      <h2>🗺️ Learning Roadmap</h2>

      <div className="roadmap-form">
        <input
          type="text"
          placeholder="Enter your learning goal"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
        />

        <select value={timePerDay} onChange={(e) => setTimePerDay(e.target.value)}>
          <option value="1">1 hour/day</option>
          <option value="2">2 hours/day</option>
          <option value="3">3 hours/day</option>
        </select>

        <select
          value={learningStyle}
          onChange={(e) => setLearningStyle(e.target.value)}
        >
          <option value="visual">Visual</option>
          <option value="auditory">Auditory</option>
          <option value="reading">Reading</option>
        </select>

        <button onClick={generateRoadmap} disabled={loading}>
          {loading ? "Generating..." : "Generate Roadmap"}
        </button>
      </div>

      {error && <div className="error-state">❌ {error}</div>}

      {roadmap && (
        <div className="roadmap-result">
          <h3>📅 Study Commitment</h3>
          <p>
            {roadmap.time_per_day}{" "}
            {roadmap.time_per_day === 1 ? "hour" : "hours"} per day for{" "}
            {roadmap.weeks} weeks
          </p>

          <h3>📆 Weekly Plan</h3>
          <ul>
            {roadmap.weekly_plan.map((week, i) => (
              <li key={i}>{week}</li>
            ))}
          </ul>

          <h3>🧠 Study Pattern</h3>
          <p>{roadmap.study_pattern}</p>

          <h3>📚 Resources</h3>
          <div className="resources-grid">
            {roadmap.resources.map((res, i) => (
              <a
                key={i}
                href={res.link}
                target="_blank"
                rel="noopener noreferrer"
                className="resource-card"
              >
                <h5>{res.title}</h5>
                <p>Click to explore →</p>
              </a>
            ))}
          </div>

          <button onClick={() => window.print()}>📄 Download / Print</button>
        </div>
      )}
    </div>
  );
}

export default Roadmap;
