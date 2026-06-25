import { useState } from "react";
import AnalysisSection from "./components/AnalysisSection";
import BackendSection from "./components/BackendSection";

export default function App() {
  const [pipelines, setPipelines] = useState([
    {
      id: "cross_modal_cnn",
      icon: "🎯",
      name: "Cross-Modal CNN Fusion",
      status: "Ready to use",
      statusColor: "#10a37f",
      file: "efficientnet_backbone.pth + audio_backbone.pth + deepfake_detector.pt",
      desc: "Extracts synchronized 12-frame facial sequences (MTCNN) and acoustic Mel-spectrograms. Fuses spatial and audio representations via a Multi-Head Cross-Attention (MHA) block to identify lip-sync mismatches.",
      params: "Trained on DFDC & FakeAVCeleb",
      device: "Local GPU / CPU Inference",
      shape: "12 visual keyframes + 79-frame acoustic tensor",
      uploaded: true
    },
    {
      id: "meta_learning",
      icon: "🔬",
      name: "Episodic Meta-Learning",
      status: "Under Development",
      statusColor: "#eab308",
      file: "meta_adapter.pth (Locked)",
      desc: "Utilizes Few-Shot Test-Time Adaptation (TTA-MAML) to dynamically adjust classifier weights, improving robustness against unseen generative models and diverse camera qualities.",
      params: "Episodic fine-tuning (R&D)",
      device: "Local GPU / CPU Inference",
      shape: "Domain-adaptive latent features",
      uploaded: true
    },
    {
      id: "llm",
      icon: "🤖",
      name: "Cognitive LLM Agent",
      status: "Under Development",
      statusColor: "#eab308",
      file: "Gemini / DeepSeek R-1 API",
      desc: "Combines localized computer vision metrics with biometrics, coordinated by an LLM reasoning agent to provide natural language explainability and cognitive fact-checking.",
      params: "Powered by Gemini 1.5 Pro API",
      device: "Cloud API Execution",
      shape: "Multi-modal contextual parsing"
    },
  ]);

  const [mode, setMode] = useState("cross_modal_cnn");
  const [selectedModel, setSelectedModel] = useState(null);

  const handleCardClick = (pipe) => {
    if (pipe.status === "Under Development") {
      alert("This detection mode is currently under development and not yet available.");
      return;
    }
    setSelectedModel(selectedModel?.id === pipe.id ? null : pipe);
    setMode(pipe.id);
  };

  const handlePromptChange = (newPrompt) => {
    setPipelines(prev => prev.map(p => {
      if (p.id === "llm") {
        return { ...p, desc: newPrompt };
      }
      return p;
    }));

    setSelectedModel(prev => {
      if (prev && prev.id === "llm") {
        return { ...prev, desc: newPrompt };
      }
      return prev;
    });
  };

  return (
    <div className="app-container" style={{ flexDirection: "column", background: "var(--bg-app)" }}>
      {/* TOP HEADER */}
      <header className="no-print" style={{ background: "var(--primary)", color: "#fff", padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
         <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
           <img src="/logo.png" alt="Forensic.AI Logo" style={{ height: "32px", width: "32px", borderRadius: "8px", objectFit: "cover" }} />
           <span style={{ fontSize: "22px", fontWeight: "600", letterSpacing: "0.5px" }}>Forensic.AI</span>
           <div style={{ display: "flex", flexDirection: "column", marginLeft: "16px", paddingLeft: "16px", borderLeft: "1px solid rgba(255,255,255,0.2)", fontSize: "11px", color: "rgba(255,255,255,0.85)", lineHeight: "1.3" }}>
             <span><strong>Developer:</strong> ETTAQAFI Ossama</span>
             <span><strong>Encadrant:</strong> Pr. KARTIT Ali</span>
           </div>
         </div>
         <div style={{ background: "#4dabf5", padding: "8px 24px", borderRadius: "12px", fontWeight: "700", fontSize: "18px" }}>Pro</div>
      </header>

      <div style={{ display: "flex", flex: 1, overflow: "hidden", background: "var(--bg-sidebar)", boxShadow: "0 4px 20px rgba(0,0,0,0.05)" }}>
        {/* LEFT SIDEBAR */}
        <aside className="sidebar no-print" style={{ borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "24px", width: "260px" }}>
          
          <div>
            {/* DETECTION METHODS */}
        <div className="sidebar-section-title">Detection Methods</div>
        <div className="sidebar-accordion">
          {pipelines.map((pipe) => {
            const isActive = selectedModel?.id === pipe.id;
            const isWorkspaceMode = mode === pipe.id;
            return (
              <div key={pipe.id} className={`pipeline-item-wrapper ${pipe.id} ${isWorkspaceMode ? "active" : ""}`}>
                <button
                  className={`sidebar-card interactive-btn ${pipe.id} ${isWorkspaceMode ? "active" : ""}`}
                  onClick={() => handleCardClick(pipe)}
                  title={pipe.status === "Under Development" ? "This mode is currently under development" : "Click to select this detection mode"}
                  style={{ opacity: pipe.status === "Under Development" ? 0.6 : 1, cursor: pipe.status === "Under Development" ? "not-allowed" : "pointer" }}
                >
                  <div className="sidebar-card-glow"></div>
                  <div className="sidebar-card-header">
                    <span className="sidebar-card-icon">{pipe.icon}</span>
                    <span className="sidebar-card-title">{pipe.name}</span>
                  </div>
                  <div className="sidebar-card-meta">
                    <span className="sidebar-card-status">
                      <span className="status-indicator-dot" style={{ backgroundColor: pipe.statusColor }}></span>
                      {pipe.status}
                    </span>
                    <span className={`sidebar-card-action ${isWorkspaceMode ? "active" : ""}`}>
                      {isWorkspaceMode ? "ACTIVE" : "REGISTRY"}
                    </span>
                  </div>
                </button>

                {isActive && (
                  <div className="model-registry-drawer animate-fade">
                    <div className="drawer-header">
                      <span className="drawer-title">
                        ℹ️ About this method
                      </span>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedModel(null);
                        }}
                        className="drawer-close-btn"
                        style={{ background: 'transparent', border: 'none', color: '#9CA3AF', fontSize: '18px', cursor: 'pointer' }}
                      >
                        ×
                      </button>
                    </div>
                    <div className="drawer-body">
                      <p className="drawer-desc">
                        {pipe.desc}
                      </p>
                      
                      {pipe.id !== "llm" ? (
                        <>
                          <div className="drawer-metric">
                            <span className="dm-label">Training data:</span>
                            <span className="dm-val">{pipe.params}</span>
                          </div>
                          <div className="drawer-metric">
                            <span className="dm-label">What it analyzes:</span>
                            <span className="dm-val font-mono">{pipe.shape}</span>
                          </div>
                          <div className="drawer-metric">
                            <span className="dm-label">Where it runs:</span>
                            <span className="dm-val">{pipe.device}</span>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="drawer-metric">
                            <span className="dm-label">How it works:</span>
                            <span className="dm-val">Reads and explains the video in plain English</span>
                          </div>
                          <div className="drawer-metric">
                            <span className="dm-label">Powered by:</span>
                            <span className="dm-val">Google Gemini AI</span>
                          </div>
                          <div className="drawer-metric">
                            <span className="dm-label">Where it runs:</span>
                            <span className="dm-val">Securely in the cloud</span>
                          </div>
                        </>
                      )}

                      {/* simulated weights drop zone / prompt editor */}
                      {pipe.id === "llm" ? (
                        <div className="prompt-editor-container">
                          <span style={{ fontSize: "11px", color: "var(--primary)", fontWeight: "600", display: "block", marginBottom: "6px" }}>
                            ✍️ Custom Analysis Prompt
                          </span>
                          <textarea
                            className="prompt-textarea"
                            value={pipe.desc}
                            onChange={(e) => handlePromptChange(e.target.value)}
                            placeholder="Enter custom prompt to analyze frames and sound..."
                          />
                        </div>
                      ) : (
                        <div style={{ marginTop: "12px", padding: "10px 12px", background: "rgba(16, 163, 127, 0.05)", border: "1px dashed rgba(16, 163, 127, 0.2)", borderRadius: "var(--radius-sm)" }}>
                          <span style={{ fontSize: "11px", color: "var(--primary)", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px" }}>
                            ✅ AI model is loaded and ready
                          </span>
                          <p style={{ fontSize: "10.5px", color: "var(--text-muted)", margin: "4px 0 0 0", lineHeight: "1.4" }}>
                            The detection model is pre-loaded on the server and ready to analyze your video immediately.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        {/* DETAILED ACCORDION DOCUMENTATION */}
        <div className="sidebar-section-title">How it works</div>
        <div className="sidebar-accordion">
          <BackendSection />
        </div>
      </div>
          
      {/* FOOTER INFO */}
      <div style={{ borderTop: "1px solid var(--border)", paddingTop: "16px", marginTop: "auto" }}>
            <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "4px" }}>Developer</div>
            <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-main)", marginBottom: "12px" }}>ETTAQAFI Ossama</div>
            <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "4px" }}>Encadrant</div>
            <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-main)" }}>Pr. KARTIT Ali</div>
          </div>
        </aside>

        {/* ========================================================
            2. MAIN WORKSPACE AREA
           ======================================================== */}
        <main className="main-workspace" style={{ flex: 1, position: "relative", overflow: "hidden", display: "flex" }}>
          {pipelines.map(pipe => (
            <div key={pipe.id} style={{ display: mode === pipe.id ? "flex" : "none", flexDirection: "column", flex: 1, width: "100%", height: "100%" }}>
              <AnalysisSection pipelines={pipelines} mode={pipe.id} />
            </div>
          ))}
        </main>
      </div>
    </div>
  );
}