export default function ResearchSection() {
  return (
    <section id="research" className="section">
      <h2>Research</h2>

      <div className="research-content">

        <p>
          This project explores modern AI-driven approaches for deepfake
          detection using face-focused forensic analysis, convolutional
          neural networks, and meta-learning architectures.
        </p>

        <p>
          The current system combines MTCNN face extraction with an
          EfficientNet-based encoder to generate facial embeddings
          capable of capturing subtle manipulation artifacts and
          synthetic inconsistencies.
        </p>

        <p>
          A prototype meta-learning framework inspired by Prototypical
          Networks is used to improve adaptability against unseen
          deepfake generation techniques and cross-dataset variations.
          Instead of relying only on fixed classification boundaries,
          the model learns representative embedding prototypes for
          real and manipulated media.
        </p>

        <p>
          The research direction also investigates future integration of
          Vision Transformers (ViTs), temporal consistency analysis,
          and multimodal reasoning systems to improve robustness,
          explainability, and generalization in real-world forensic scenarios.
        </p>

      </div>
    </section>
  );
}