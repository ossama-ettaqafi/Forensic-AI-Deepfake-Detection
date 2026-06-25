import ResultCard from "./ResultCard";

export default function ResultsSection({ result }) {
  // Safety guard: prevent broken objects from crashing UI
  const safeResult =
    result && typeof result === "object"
      ? result
      : null;

  return (
    <section id="results" className="section">
      <h2>Results</h2>

      {!safeResult ? (
        <p>No analysis yet</p>
      ) : (
        <ResultCard result={safeResult} />
      )}
    </section>
  );
}