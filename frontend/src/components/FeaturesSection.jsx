export default function FeaturesSection() {
  return (
    <section id="features" className="section">
      <h2>Features</h2>

      <div className="features-grid">

        <div className="feature-box">
          <span className="feature-icon">🎥</span>
          <span>Video Detection</span>
        </div>

        <div className="feature-box">
          <span className="feature-icon">🖼️</span>
          <span>Image Analysis</span>
        </div>

        <div className="feature-box">
          <span className="feature-icon">🧠</span>
          <span>AI Prediction</span>
        </div>

        <div className="feature-box">
          <span className="feature-icon">📊</span>
          <span>Confidence Score</span>
        </div>

      </div>
    </section>
  );
}