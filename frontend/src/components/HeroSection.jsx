export default function HeroSection() {
  const scrollToAnalysis = () => {
    const section = document.getElementById("analysis");
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <section className="hero">
      <h1>AI Deepfake Detection</h1>
      <p>Analyze media using multi-model AI systems</p>

      <button onClick={scrollToAnalysis}>
        Start Analysis
      </button>
    </section>
  );
}