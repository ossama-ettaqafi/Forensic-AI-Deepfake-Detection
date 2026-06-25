import { useState } from "react";

const ITEMS = [
  {
    icon: "🧬",
    title: "1. Dual CNN Feature Extraction",
    content: "The core engine simultaneously analyzes spatial visual artifacts (like pixel blending on the face) and acoustic anomalies (like synthetic vocal frequencies) using independent Convolutional Neural Networks."
  },
  {
    icon: "⚡",
    title: "2. Early-Exit Optimization",
    content: "If the visual CNN detects undeniable, high-confidence manipulation artifacts early in the pipeline, it can optionally skip downstream acoustic processing to deliver instant results."
  },
  {
    icon: "👄",
    title: "3. Multi-Head Attention Lip-Sync",
    content: "A specialized cross-modal attention mechanism compares mouth movements directly with phonetic audio tracks. It mathematically flags temporal mismatches between what is spoken and how the lips move."
  },
  {
    icon: "⚖️",
    title: "4. Cognitive AI Reports",
    content: "Raw analytical tensors and confidence logits are fed into Google Gemini 2.5 Flash, which synthesizes the data into a human-readable digital forensic report, explaining exactly why the media was flagged."
  }
];

export default function BackendSection() {
  const [openIndex, setOpenIndex] = useState(0);

  const toggle = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="accordion-container">
      {ITEMS.map((item, index) => (
        <div
          key={index}
          className={`accordion-card ${
            openIndex === index ? "active" : ""
          }`}
        >
          <button
            className="accordion-header"
            onClick={() => toggle(index)}
          >
            <div className="accordion-title">
              <span className="accordion-emoji">
                {item.icon}
              </span>
              <span>{item.title}</span>
            </div>
            <span className="accordion-toggle">
              {openIndex === index ? "−" : "+"}
            </span>
          </button>
          {openIndex === index && (
            <div className="accordion-content">
              <p>{item.content}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}