export default function DocumentationSection() {
  return (
    <section id="docs" className="section">
      <h2>System Documentation</h2>

      <div className="doc-grid">

        <div className="doc-card">
          <h3>🧠 4-Mode Architecture</h3>
          <p>
            Forensic.AI is structured as an advanced 4-mode deepfake detection framework:
          </p>
          <ul style={{ fontSize: "12.5px", color: "var(--text-muted)", listStyleType: "none", paddingLeft: 0, marginTop: "8px", display: "flex", flexDirection: "column", gap: "6px" }}>
            <li><strong style={{ color: "var(--success)" }}>🟢 Mode 1: Cross-Modal CNN</strong> (Active & Calibrated)</li>
            <li><strong style={{ color: "var(--primary)" }}>🟡 Mode 2: Episodic Meta-Learning</strong> (Under Development)</li>
            <li><strong style={{ color: "var(--primary)" }}>🟡 Mode 3: Multi-Agent LLM Reasoning ([42])</strong> (Under Development)</li>
            <li><strong style={{ color: "var(--primary)" }}>🟡 Mode 4: Combined Forensics Consensus</strong> (Under Development)</li>
          </ul>
        </div>

        <div className="doc-card">
          <h3>⚙️ Mode 1: How It Works</h3>
          <p>
            The active <strong>Cross-Modal CNN</strong> extracts 12 facial keyframes (MTCNN) and standardizes them for spatial-temporal Convolutions (3D-CNN). Synchronously, the audio is converted to Log-Mel Spectrogram matrices and parsed via an acoustic 1D-CNN. A Multi-Head Cross-Attention (MHA) Transformer correlates lip visemes and speech phonemes.
          </p>
        </div>

        <div className="doc-card">
          <h3>📊 Result Generation</h3>
          <p>
            For the active Mode 1, our visual and acoustic branches yield separate probability scores, which are joined at the decision level through late fusion. The Sync Correlator analyzes Shannon entropy across cross-attention weights. Mismatches > 80ms trigger immediate synthetic anomalies alerts.
          </p>
        </div>

        <div className="doc-card">
          <h3>⚠️ Limitations & Codecs</h3>
          <p>
            Highly compressed web media or extremely low-resolution face targets can cause domain-shift errors. This will be resolved by Mode 2 (TTA-MAML Test-Time adaptation), which is currently undergoing weights calibration at our backend registry.
          </p>
        </div>

        <div className="doc-card">
          <h3>🔬 Research Basis ([42])</h3>
          <p>
            The framework aligns directly with the multi-modal taxomatic verrous of deepfake forensic literature (IEEE Access 2026). Our planned Mode 3 multi-agent architecture incorporates local computer vision (YOLOv8 + FaceNet) and audio biometrics (Whisper) with reasoning models like DeepSeek R-1.
          </p>
        </div>

        <div className="doc-card">
          <h3>📁 Media Specifications</h3>
          <p>
            Enforces strict multi-modal parsing. Uploads must be video formats (MP4, AVI, MOV, MKV) that contain both high-resolution visual tracks and continuous voice tracks for sync correlation.
          </p>
        </div>

      </div>
    </section>
  );
}