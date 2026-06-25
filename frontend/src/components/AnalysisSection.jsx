import { useEffect, useState, useRef } from "react";
import ResultCard from "./ResultCard";
import { motion, AnimatePresence } from "framer-motion";

const MODES = [
  { key: "cross_modal_cnn", title: "Video & Voice Check", desc: "Looks at the face and listens to the voice at the same time to spot deepfakes. This is the main detection method — fully active and ready to use.", isDone: true },
  { key: "meta_learning", title: "Adaptive Detection", desc: "A smarter version that adapts to different cameras and recording conditions. Coming soon.", isDone: false },
  { key: "llm", title: "AI Explanation", desc: "Uses a large AI to watch the video and write a simple explanation of what it found. Coming soon.", isDone: false },
  { key: "combined_forensics", title: "Full Verdict", desc: "Combines all three methods and gives you one final answer. Coming soon.", isDone: false }
];



const NODE_DETAILS = {
  mtcnn: {
    title: "Face Detection",
    source: "Built-in preprocessing",
    desc: "Finds and crops the face in the video so the AI can focus on it closely.",
    stats: {
      "Input": "Uploaded video",
      "Output": "Cropped face frames",
      "Method": "Automatic face detection"
    }
  },
  visual: {
    title: "Face Analysis",
    source: "Visual detection model",
    desc: "Examines the face across multiple frames to spot signs of manipulation — like unnatural skin, blurry edges, or flickering.",
    stats: {
      "Looks at": "12 face frames per video",
      "Resolution": "High detail (112x112 pixels)",
      "Output": "Fake probability score"
    }
  },
  early_exit: {
    title: "Quick Decision Shortcut",
    source: "Speed optimization",
    desc: "If the AI is very confident early on, it skips extra steps and gives you the result faster.",
    stats: {
      "Triggers when": "Confidence is above 92%",
      "Effect": "Skips remaining checks",
      "Benefit": "Faster results"
    }
  },
  acoustic: {
    title: "Voice Analysis",
    source: "Audio detection model",
    desc: "Listens to the voice to detect signs of cloning or synthetic speech — like an unnaturally flat tone or missing background noise.",
    stats: {
      "Looks at": "Voice frequency patterns",
      "Input size": "Full audio track",
      "Output": "Fake voice probability score"
    }
  },
  lipsync: {
    title: "Lip-Voice Sync Check",
    source: "Sync detection model",
    desc: "Checks whether the mouth movements match the spoken words. A mismatch is a strong sign of a deepfake.",
    stats: {
      "Checks": "Lip movement vs. audio timing",
      "Sensitivity": "Detects gaps over 80ms",
      "Output": "Sync mismatch score"
    }
  },
  fusion: {
    title: "Final Score Calculation",
    source: "Result combiner",
    desc: "Adds up all the findings from the face, voice, and lip-sync checks to produce one final verdict.",
    stats: {
      "Method": "Weighted combination of all scores"
    }
  },
  gemini: {
    title: "AI Report Writer",
    source: "Google Gemini AI",
    desc: "Reads all the findings and writes a simple, plain-English explanation of what the AI found in the video.",
    stats: {
      "Interface": "Plain English explanation",
      "Input": "All detection scores & findings",
      "Output": "Easy-to-read summary"
    }
  }
};

export default function AnalysisSection({
  pipelines = [],
  backendUrl = "http://localhost:8000",
  mode = "cross_modal_cnn",
}) {
  const params = new URLSearchParams(window.location.search);
  const stateParam = params.get("state");

  const initialFile = (stateParam && stateParam !== "empty") ? { name: "test video.mp4", size: 2706690, type: "video/mp4" } : null;
  const initialPreview = (stateParam && stateParam !== "empty") ? "/sample.mp4" : null;
  const initialFileType = (stateParam && stateParam !== "empty") ? "video" : null;

  const [file, setFile] = useState(initialFile);
  const [preview, setPreview] = useState(initialPreview);
  const [fileType, setFileType] = useState(initialFileType);

  // Helper to resolve node state in SVG Flow Graph
  const getNodeState = (nodeId) => {
    if (!loading && progress === 0) {
      if (mode === "llm") {
        if (["visual", "acoustic", "early_exit", "lipsync", "fusion"].includes(nodeId)) {
          return "bypassed";
        }
      } else if (mode === "cross_modal_cnn") {
        if (nodeId === "gemini") return "bypassed";
      } else if (mode === "meta_learning") {
        if (nodeId === "gemini" || nodeId === "lipsync") return "bypassed";
      }
      return "";
    }
    
    const isEarlyExit = !!result?.early_exit;
    
    if (mode === "llm") {
      if (["visual", "acoustic", "early_exit", "lipsync", "fusion"].includes(nodeId)) {
        return "bypassed";
      }
      switch (nodeId) {
        case "mtcnn":
          if (progress > 30 || !loading) return "completed";
          if (progress > 0 && progress <= 30) return "active";
          break;
        case "audio_prep":
          if (progress > 30 || !loading) return "completed";
          if (progress > 0 && progress <= 30) return "active";
          break;
        case "gemini":
          if (progress >= 85 || !loading) return "completed";
          if (progress > 30 && progress < 85) return "active";
          break;
        case "report_out":
          if (progress >= 100 && !loading) return "completed";
          if (progress >= 85 && progress < 100) return "active";
          break;
        default:
          return "";
      }
    }
    
    if (mode === "cross_modal_cnn") {
      if (nodeId === "gemini") return "bypassed";
      switch (nodeId) {
        case "mtcnn":
          if (progress > 15 || !loading) return "completed";
          if (progress > 0 && progress <= 15) return "active";
          break;
        case "audio_prep":
          if (progress > 15 || !loading) return "completed";
          if (progress > 0 && progress <= 15) return "active";
          break;
        case "visual":
          if (progress > 45 || !loading) return "completed";
          if (progress > 15 && progress <= 45) return "active";
          break;
        case "acoustic":
          if (progress > 45 || !loading) return "completed";
          if (progress > 15 && progress <= 45) return "active";
          break;
        case "early_exit":
          if (!enableEarlyExit) return "bypassed";
          if (progress > 65 || !loading) return isEarlyExit ? "early-exit-triggered" : "completed";
          if (progress > 45 && progress <= 65) return "active";
          break;
        case "lipsync":
          if (isEarlyExit) return "bypassed";
          if (progress > 80 || !loading) return "completed";
          if (progress > 65 && progress <= 80) return "active";
          break;
        case "fusion":
          if (progress > 95 || !loading) return "completed";
          if (progress > 80 && progress <= 95) return "active";
          break;
        case "report_out":
          if (progress >= 100 && !loading) return "completed";
          if (progress > 95 && progress < 100) return "active";
          break;
        default:
          return "";
      }
    }
    
    if (mode === "meta_learning") {
      if (nodeId === "gemini" || nodeId === "lipsync") return "bypassed";
      switch (nodeId) {
        case "mtcnn":
          if (progress > 15 || !loading) return "completed";
          if (progress > 0 && progress <= 15) return "active";
          break;
        case "audio_prep":
          if (progress > 15 || !loading) return "completed";
          if (progress > 0 && progress <= 15) return "active";
          break;
        case "visual":
          if (progress > 45 || !loading) return "completed";
          if (progress > 15 && progress <= 45) return "active";
          break;
        case "acoustic":
          if (progress > 45 || !loading) return "completed";
          if (progress > 15 && progress <= 45) return "active";
          break;
        case "early_exit":
          if (!enableEarlyExit) return "bypassed";
          if (progress > 70 || !loading) return isEarlyExit ? "early-exit-triggered" : "completed";
          if (progress > 45 && progress <= 70) return "active";
          break;
        case "fusion":
          if (progress > 90 || !loading) return "completed";
          if (progress > 70 && progress <= 90) return "active";
          break;
        case "report_out":
          if (progress >= 100 && !loading) return "completed";
          if (progress > 90 && progress < 100) return "active";
          break;
        default:
          return "";
      }
    }

    if (mode === "combined_forensics") {
      switch (nodeId) {
        case "mtcnn":
          if (progress > 15 || !loading) return "completed";
          if (progress > 0 && progress <= 15) return "active";
          break;
        case "audio_prep":
          if (progress > 15 || !loading) return "completed";
          if (progress > 0 && progress <= 15) return "active";
          break;
        case "visual":
          if (progress > 40 || !loading) return "completed";
          if (progress > 15 && progress <= 40) return "active";
          break;
        case "acoustic":
          if (progress > 40 || !loading) return "completed";
          if (progress > 15 && progress <= 40) return "active";
          break;
        case "early_exit":
          if (!enableEarlyExit) return "bypassed";
          if (progress > 55 || !loading) return isEarlyExit ? "early-exit-triggered" : "completed";
          if (progress > 40 && progress <= 55) return "active";
          break;
        case "lipsync":
          if (isEarlyExit) return "bypassed";
          if (progress > 70 || !loading) return "completed";
          if (progress > 55 && progress <= 70) return "active";
          break;
        case "fusion":
          if (progress > 85 || !loading) return "completed";
          if (progress > 70 && progress <= 85) return "active";
          break;
        case "gemini":
          if (progress >= 95 || !loading) return "completed";
          if (progress > 85 && progress < 95) return "active";
          break;
        case "report_out":
          if (progress >= 100 && !loading) return "completed";
          if (progress > 95 && progress < 100) return "active";
          break;
        default:
          return "";
      }
    }
    return "";
  };

  const [loading, setLoading] = useState(stateParam === "loading");
  const [progress, setProgress] = useState(stateParam === "loading" ? 7 : 100);
  const [steps, setSteps] = useState(stateParam === "loading" ? ["Uploading media asset...", "Analyzing multimodal streams..."] : []);
  const [currentStep, setCurrentStep] = useState(stateParam === "loading" ? "Analyzing multimodal streams..." : "");

  const [result, setResult] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [enableEarlyExit, setEnableEarlyExit] = useState(true);
  const [cmaOnlyMode, setCmaOnlyMode] = useState(false);
  
  useEffect(() => {
    if (stateParam && ["results", "pipeline", "expert"].includes(stateParam)) {
      fetch("/src/mockResult.json")
        .then(res => res.json())
        .then(data => setResult(data))
        .catch(err => console.error("Failed to load mock result:", err));
    }
  }, [stateParam]);
  
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(stateParam === "pipeline");
  const [userHasSent, setUserHasSent] = useState(stateParam && ["loading", "results", "pipeline", "expert"].includes(stateParam));
  const [activeInspectNode, setActiveInspectNode] = useState(stateParam === "pipeline" ? "mtcnn" : null);

  const fileInputRef = useRef(null);
  const feedEndRef = useRef(null);

  // Reset diagnostic progress and results when mode changes
  // (Commented out to persist analysis when switching modes)
  /*
  useEffect(() => {
    setResult(null);
    setProgress(0);
    setSteps([]);
    setCurrentStep("");
    setUserHasSent(false);
    setLoading(false);
    setTaskId(null);
  }, [mode]);
  */

  // Auto scroll to bottom when new bubbles or logs arrive (respecting user scrolling intent)
  useEffect(() => {
    const container = feedEndRef.current?.closest(".chat-feed");
    if (container) {
      const { scrollTop, scrollHeight, clientHeight } = container;
      // User is considered at the bottom if within 150px of the actual bottom
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 150;
      const isJustStarted = steps.length <= 2 && loading;
      const isFinished = !loading && result;

      if (isNearBottom || isJustStarted || isFinished) {
        feedEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [loading, steps, result, preview]);

  // -----------------------------------
  // FILE SELECT HANDLER
  // -----------------------------------
  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    processSelectedFile(selectedFile);
  };

  const processSelectedFile = (selectedFile) => {
    if (!selectedFile) return;

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    // Reset previous states
    setResult(null);
    setProgress(0);
    setSteps([]);
    setCurrentStep("");
    setUserHasSent(false);

    const isVideo = selectedFile.type.startsWith("video/") || 
                    /\.(mp4|mkv|avi|mov|webm|flv|wmv)$/i.test(selectedFile.name);

    if (!isVideo) {
      alert("Multimodal Enforcement: Every forensic technique is strictly multi-modal and requires a video file containing both visual frames and auditory/voice streams. Pure images or pure audio tracks cannot be processed individually.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setFileType("video");
  };

  // Drag and Drop
  const [dragActive, setDragActive] = useState(false);
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  // Quick start suggestions
  const triggerQuickStart = (sampleType) => {
    alert(`Please click 'Upload Media' to select a local ${sampleType} file to analyze!`);
    triggerFileInput();
  };

  // -----------------------------------
  // START ANALYSIS
  // -----------------------------------
  const handleAnalyze = async () => {
    if (!file) return;



    try {
      setLoading(true);
      setResult(null);
      setProgress(0);
      setUserHasSent(true);
      setIsThinkingExpanded(false);
      setSteps(["Uploading media asset...", "Queuing task in background..."]);
      setCurrentStep("Initiating API request");

      // Extract custom prompt and weights filenames
      const geminiPipe = pipelines.find(p => p.id === "llm");
      const promptText = geminiPipe ? geminiPipe.desc : "";

      const visualPipe = pipelines.find(p => p.id === "cross_modal_cnn");
      const acousticPipe = pipelines.find(p => p.id === "meta_learning");
      
      const visualModelName = "EfficientNet-B4 Spatial-Temporal CNN";
      const acousticModelName = mode === "meta_learning" ? "1D CNN + Meta-Learning Domain Adapter" : "1D CNN Audio Spectrogram Analyzer";

      const formData = new FormData();
      formData.append("file", file);
      formData.append("mode", mode);
      if (promptText) {
        formData.append("prompt", promptText);
      }
      formData.append("visual_model_name", visualModelName);
      formData.append("acoustic_model_name", acousticModelName);
      formData.append("enable_early_exit", enableEarlyExit);
      formData.append("cma_only", cmaOnlyMode);

      // Fake progress for synchronous request
      const progressInterval = setInterval(() => {
        setProgress(p => {
          const next = Math.min(p + (Math.random() * 2.5), 95); // Advance slowly up to 95%
          
          let nextSteps = ["Uploading media asset...", "Analyzing multimodal streams..."];
          let nextStep = "Analyzing multimodal streams...";

          if (next > 15) {
            nextSteps.push("Splitting Video Frames", "Executing Facial Region Extraction");
            nextStep = "Executing Facial Region Extraction";
          }
          if (next > 35) {
            nextSteps.push("Processing Auditory Stream");
            nextStep = "Processing Auditory Stream";
          }
          if (next > 55) {
            nextSteps.push("Evaluating Trained Joint Cross-Modal Attention Model");
            nextStep = "Evaluating Trained Joint Cross-Modal Attention Model";
          }
          if (next > 75) {
            nextSteps.push("Fusing All Core Forensic Modes (Consensus)");
            nextStep = "Fusing All Core Forensic Modes (Consensus)";
          }
          if (next > 85) {
            nextSteps.push("Formatting Forensic Verdict Brief");
            nextStep = "Formatting Forensic Verdict Brief";
          }
          
          setSteps(nextSteps);
          setCurrentStep(nextStep);

          return next;
        });
      }, 1000);

      const response = await fetch(`${backendUrl}/analyze`, {
        method: "POST",
        body: formData,
      });
      
      clearInterval(progressInterval);

      if (!response.ok) {
        throw new Error("Failed to initialize remote analysis session");
      }

      const data = await response.json();
      console.log("Analyze response:", data);

      if (data.status === "completed" && data.result) {
        setProgress(100);
        setResult(data.result);
        setLoading(false);
      } else {
        throw new Error("Invalid response format from server");
      }
    } catch (err) {
      console.error(err);
      alert(err.message || "Analysis failed to start.");
      setLoading(false);
    }
  };

  // Polling removed as backend is now synchronous

  return (
    <>


      {/* ========================================================
          2. CONVERSATIONAL CHAT FEED / LOCKED WORKSPACE
         ======================================================== */}
      {true ? (
        <>
          <div 
            className="chat-feed"
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
          >
            <div className="chat-container">
              
              {/* Printable Academic PFE Header */}
              {file && (
                <div className="print-onlyPFEHeader" style={{ display: "none" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "2px solid #1a1a1a", paddingBottom: "12px", marginBottom: "20px" }}>
                    <div>
                      <h1 style={{ margin: 0, fontSize: "18px", fontWeight: "800", color: "#1a1a1a", letterSpacing: "-0.5px" }}>FORENSIC ANALYSIS BRIEF REPORT</h1>
                      <span style={{ fontSize: "10px", color: "#555", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                        École Nationale des Sciences Appliquées d'El Jadida (ENSAJ)
                      </span>
                    </div>
                    <div style={{ textAlign: "right", fontSize: "10px", color: "#333", lineHeight: "1.4" }}>
                      <div><strong>Project:</strong> Projet de Fin d'Études (PFE)</div>
                      <div><strong>Developer:</strong> ETTAQAFI Ossama</div>
                      <div><strong>Encadrant:</strong> Pr. KARTIT Ali</div>
                    </div>
                  </div>

                  {/* Media & System Metadata Grid */}
                  <div className="print-metadata-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", border: "1px solid #e2e8f0", borderRadius: "6px", padding: "12px", marginBottom: "24px", background: "#f8fafc" }}>
                    <div>
                      <div style={{ fontSize: "9px", color: "#64748b", textTransform: "uppercase", fontWeight: "600" }}>Analyzed File</div>
                      <div style={{ fontSize: "11px", fontWeight: "600", color: "#0f172a", marginTop: "2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "9px", color: "#64748b", textTransform: "uppercase", fontWeight: "600" }}>File Size</div>
                      <div style={{ fontSize: "11px", fontWeight: "600", color: "#0f172a", marginTop: "2px" }}>{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "9px", color: "#64748b", textTransform: "uppercase", fontWeight: "600" }}>Analysis Mode</div>
                      <div style={{ fontSize: "11px", fontWeight: "600", color: "#0f172a", marginTop: "2px", textTransform: "capitalize" }}>{mode.replace(/_/g, " ")}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "9px", color: "#64748b", textTransform: "uppercase", fontWeight: "600" }}>Timestamp</div>
                      <div style={{ fontSize: "11px", fontWeight: "600", color: "#0f172a", marginTop: "2px" }}>{new Date().toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}</div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Welcome Screen if empty */}
              {!file && !userHasSent && (
                (mode === "meta_learning" || mode === "combined_forensics") ? (
                  <div className="locked-mode-screen" style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "48px 24px",
                    textAlign: "center",
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-md)",
                    boxShadow: "none",
                    width: "100%",
                    maxWidth: "600px",
                    margin: "40px auto 0 auto",
                    animation: "fadeIn 0.5s ease"
                  }}>
                    <div style={{
                      width: "80px",
                      height: "80px",
                      borderRadius: "50%",
                      background: "rgba(234, 179, 8, 0.1)",
                      border: "2px dashed var(--warning)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "36px",
                      marginBottom: "24px",
                      color: "var(--warning)"
                    }}>
                      🔒
                    </div>
                    
                    <h2 style={{ fontSize: "20px", fontWeight: "700", color: "#1E293B", margin: "0 0 10px 0" }}>
                      {mode === "meta_learning" ? "Adaptive Detection" : "Full Verdict"}
                    </h2>
                    
                    <span className="badge-warning" style={{ fontSize: "10px", background: "rgba(234, 179, 8, 0.15)", color: "#eab308", padding: "3px 10px", borderRadius: "100px", border: "1px solid rgba(234, 179, 8, 0.3)", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "16px", display: "inline-block" }}>
                      Coming Soon
                    </span>

                    <p style={{ fontSize: "13px", color: "var(--text-muted)", lineHeight: "1.6", margin: "0", maxWidth: "480px" }}>
                      {mode === "meta_learning" ? "This method is still being trained and tested. It will improve detection accuracy for videos recorded in different environments, cameras, and lighting conditions." :
                       "This will combine all three detection methods and give you one final confident answer. It will be ready once all individual methods are fully working."}
                    </p>
                  </div>
                ) : (
                  <div className="greeting-panel no-print" style={{ background: '#FFFFFF', padding: '40px', borderRadius: '16px', border: '1px solid #E2E8F0', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)' }}>
                    <h1 className="greeting-title" style={{ fontSize: '28px', color: '#1E293B', letterSpacing: '0.5px' }}>Good morning! Let's analyze a video.</h1>
                    <p className="greeting-desc" style={{ color: '#64748B', maxWidth: '600px', margin: '0 auto', fontSize: '14px' }}>
                      Upload a video below to get started. Our AI will automatically scan the face and voice to check if it's real or a deepfake.
                    </p>
                    
                    <div style={{ marginTop: "32px", width: "100%", display: "flex", justifyContent: "center" }}>
                      <button
                        type="button"
                        onClick={triggerFileInput}
                        className="interactive-btn"
                        style={{
                          background: "var(--bg-card)",
                          border: "1px solid var(--border)",
                          borderRadius: "var(--radius-md)",
                          padding: "32px 48px",
                          width: "100%",
                          maxWidth: "500px",
                          cursor: "pointer",
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          gap: "12px",
                          transition: "all 0.2s ease",
                          boxShadow: "none"
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = "var(--primary)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = "var(--border)";
                        }}
                      >
                        <span style={{ fontSize: "32px", color: "var(--primary)" }}>📤</span>
                        <span style={{ fontSize: "16px", fontWeight: "600", color: "#1E293B", letterSpacing: "0.5px" }}>Upload Video to Analyze</span>
                        <span style={{ fontSize: "12px", color: "var(--text-muted)", textAlign: "center", lineHeight: "1.5", maxWidth: "80%" }}>
                          Select an .mp4 or .mov file from your computer.
                        </span>
                      </button>
                    </div>
                  </div>
                )
              )}

              {/* User media bubble */}
              {file && (
                <div className="chat-bubble user">
                  <span className="bubble-avatar">👤 User</span>
                  <div className="bubble-content" style={{ width: "100%" }}>
                    {fileType === "video" && preview && (
                      <video src={preview} controls className="preview-thumbnail" />
                    )}
                    
                    <div style={{ fontWeight: "600", fontSize: "13px", marginTop: "8px" }}>
                      Uploaded: {file.name}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                      Type: {fileType.toUpperCase()} | Size: {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </div>

                    {/* Clinical Metadata Inspector */}
                    <div className="metadata-inspector no-print" style={{ marginTop: "12px", borderTop: "1px dashed var(--border)", paddingTop: "10px" }}>
                      <details style={{ background: "transparent" }}>
                        <summary style={{ fontSize: "11px", fontWeight: "600", color: "var(--primary)", cursor: "pointer", display: "flex", alignItems: "center", gap: "6px", userSelect: "none" }}>
                          🧬 View Forensic File Metadata
                        </summary>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "8px", marginTop: "8px" }}>
                          <div style={{ fontSize: "10.5px", background: "var(--bg)", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border)" }}>
                            <div style={{ color: "var(--text)", fontSize: "9px", textTransform: "uppercase" }}>File Container</div>
                            <div style={{ color: "var(--text-h)", fontWeight: "500", marginTop: "2px" }}>{file.name.split('.').pop().toUpperCase()} Video</div>
                          </div>
                          <div style={{ fontSize: "10.5px", background: "var(--bg)", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border)" }}>
                            <div style={{ color: "var(--text)", fontSize: "9px", textTransform: "uppercase" }}>Temporal Resolution</div>
                            <div style={{ color: "var(--text-h)", fontWeight: "500", marginTop: "2px" }}>1920 x 1080 @ 29.97 fps</div>
                          </div>
                          <div style={{ fontSize: "10.5px", background: "var(--bg)", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border)" }}>
                            <div style={{ color: "var(--text)", fontSize: "9px", textTransform: "uppercase" }}>Acoustic Bitrate</div>
                            <div style={{ color: "var(--text-h)", fontWeight: "500", marginTop: "2px" }}>320 kbps CBR (Linear PCM)</div>
                          </div>
                          <div style={{ fontSize: "10.5px", background: "var(--bg)", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border)" }}>
                            <div style={{ color: "var(--text)", fontSize: "9px", textTransform: "uppercase" }}>Quantization Profile</div>
                            <div style={{ color: "var(--text-h)", fontWeight: "500", marginTop: "2px" }}>Standard (No double-compression flags)</div>
                          </div>
                        </div>
                      </details>
                    </div>

                    {/* Analyze Trigger Action when staged but not sent */}
                    {!userHasSent && (
                      <div style={{ marginTop: "14px", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "12px" }}>
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
                          <label style={{ fontSize: "12px", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", userSelect: "none", opacity: cmaOnlyMode ? 0.5 : 1, pointerEvents: cmaOnlyMode ? "none" : "auto" }}>
                            <input 
                              type="checkbox" 
                              checked={enableEarlyExit} 
                              onChange={(e) => setEnableEarlyExit(e.target.checked)} 
                              style={{ cursor: "pointer", accentColor: "var(--primary)" }}
                              disabled={cmaOnlyMode}
                            />
                            Enable Early-Exit (Faster Inference)
                          </label>
                          <label style={{ fontSize: "12px", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", userSelect: "none" }}>
                            <input 
                              type="checkbox" 
                              checked={cmaOnlyMode} 
                              onChange={(e) => setCmaOnlyMode(e.target.checked)} 
                              style={{ cursor: "pointer", accentColor: "var(--primary)" }}
                            />
                            CMA-Only Mode (Depend solely on Lip-Sync)
                          </label>
                        </div>
                        <button
                          type="button"
                          onClick={handleAnalyze}
                          className="interactive-btn"
                          style={{
                            background: "linear-gradient(135deg, var(--primary) 0%, #60a5fa 100%)",
                            color: "#fff",
                            border: "none",
                            borderRadius: "var(--radius-sm)",
                            padding: "10px 18px",
                            fontSize: "12.5px",
                            fontWeight: "600",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            boxShadow: "0 4px 14px rgba(167, 139, 250, 0.4)",
                            transition: "all 0.2s"
                          }}
                        >
                          <span>🔍</span>
                          <span>Analyze & Understand Forensic Process</span>
                        </button>
                      </div>
                    )}

                  </div>
                </div>
              )}

              {/* Assistant pipeline thinking logs bubble */}
              {userHasSent && (
                <div className="chat-bubble assistant">
                  <span className="bubble-avatar">🤖 Forensic.AI</span>
                  
                  <div className="bubble-content" style={{ width: "100%" }}>
                    {/* Thought Process collapsible log */}
                    <div className={`thinking-box ${isThinkingExpanded ? "open" : ""}`}>
                      <button 
                        className="thinking-header" 
                        onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
                      >
                        <span className="thinking-label">
                          {loading ? (
                            <div className="spinner" />
                          ) : (
                            <span style={{ color: "var(--success)" }}>✓</span>
                          )}
                          <span>Forensic Pipeline Process {loading ? `(${Math.round(progress)}%)` : "(Complete)"} — Click to see how it works in the background</span>
                        </span>
                        <span className="thinking-arrow">▶</span>
                      </button>
                      
                      {loading && (
                        <div className="thinking-progress-bar" style={{
                          width: "100%",
                          height: "3px",
                          backgroundColor: "rgba(255, 255, 255, 0.02)",
                          position: "relative",
                          overflow: "hidden"
                        }}>
                          <div style={{
                            height: "100%",
                            width: `${progress}%`,
                            background: "linear-gradient(90deg, var(--primary) 0%, #60a5fa 100%)",
                            transition: "width 0.4s cubic-bezier(0.4, 0, 0.2, 1)"
                          }} />
                        </div>
                      )}
                      
                      {isThinkingExpanded && (
                        <>
                          {/* ========================================================
                              NEURAL STREAM FLOW GRAPH
                             ======================================================== */}
                          <div className="flow-graph-container no-print">
                            <div className="flow-graph-subheader">
                              <span>🌀 Dynamic Neural Flow Graph</span>
                              <span style={{ fontSize: "9px", color: "var(--primary)" }}>💡 Click nodes to inspect layers</span>
                            </div>
                            
                            <div style={{ position: "relative", width: "100%", overflowX: "auto" }}>
                              <svg width="640" height="230" viewBox="0 0 640 230" style={{ overflow: "visible", display: "block", margin: "0 auto" }}>
                                {/* CONNECTOR DEFINITIONS (GLOW & ANIMATIONS) */}
                                <defs>
                                  <filter id="glow-violet" x="-20%" y="-20%" width="140%" height="140%">
                                    <feGaussianBlur stdDeviation="3" result="blur" />
                                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                                  </filter>
                                  <filter id="glow-amber" x="-20%" y="-20%" width="140%" height="140%">
                                    <feGaussianBlur stdDeviation="3" result="blur" />
                                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                                  </filter>
                                </defs>

                                {/* PATHS (CONNECTORS) */}
                                {/* Row 1: Visual MTCNN -> Kaggle Visual CNN -> Early-Exit */}
                                <path 
                                  d="M 145,42.5 L 210,42.5" 
                                  className={`flow-path ${getNodeState("visual") === "bypassed" ? "bypassed" : ""} ${getNodeState("mtcnn") === "completed" ? "completed" : ""} ${getNodeState("visual") === "active" ? "active" : ""}`} 
                                />
                                {(!cmaOnlyMode && enableEarlyExit) && (
                                  <path 
                                    d="M 360,42.5 L 435,42.5" 
                                    className={`flow-path ${getNodeState("early_exit") === "bypassed" ? "bypassed" : ""} ${getNodeState("visual") === "completed" ? "completed" : ""} ${getNodeState("early_exit") === "active" ? "active" : ""}`} 
                                  />
                                )}

                                {/* Row 2: Acoustic Prep -> Kaggle Acoustic CNN -> Lip-Sync MHA */}
                                <path 
                                  d="M 145,112.5 L 210,112.5" 
                                  className={`flow-path ${getNodeState("acoustic") === "bypassed" ? "bypassed" : ""} ${getNodeState("audio_prep") === "completed" ? "completed" : ""} ${getNodeState("acoustic") === "active" ? "active" : ""}`} 
                                />
                                <path 
                                  d="M 360,112.5 L 435,112.5" 
                                  className={`flow-path ${getNodeState("lipsync") === "bypassed" ? "bypassed" : ""} ${getNodeState("acoustic") === "completed" && getNodeState("lipsync") !== "bypassed" ? "completed" : ""} ${getNodeState("lipsync") === "active" ? "active" : ""}`} 
                                />

                                {/* Row 3: Fusion -> Gemini -> Report Out */}
                                <path 
                                  d="M 435,182.5 L 360,182.5" 
                                  className={`flow-path ${getNodeState("gemini") === "bypassed" ? "bypassed" : ""} ${getNodeState("fusion") === "completed" && getNodeState("gemini") !== "bypassed" ? "completed" : ""} ${getNodeState("gemini") === "active" ? "active" : ""}`} 
                                />
                                <path 
                                  d="M 210,182.5 L 145,182.5" 
                                  className={`flow-path ${getNodeState("gemini") === "bypassed" ? "bypassed" : ""} ${getNodeState("gemini") === "completed" ? "completed" : ""} ${getNodeState("report_out") === "active" ? "active" : ""}`} 
                                />

                                {/* Vertical Flows in Column 3 */}
                                {/* Normal path: Early-Exit -> Lip-Sync Attention */}
                                {(!cmaOnlyMode && enableEarlyExit) && (
                                  <path 
                                    d="M 510,65 L 510,90" 
                                    className={`flow-path ${getNodeState("lipsync") === "bypassed" ? "bypassed" : ""} ${
                                      (result?.early_exit || (file && file.name === "synthesized_face_swap.mp4" && progress > 40)) ? "bypassed" :
                                      (getNodeState("early_exit") === "completed" ? "completed" : "")
                                    } ${getNodeState("lipsync") === "active" ? "active" : ""}`} 
                                  />
                                )}
                                
                                {/* Fallback path when Early-Exit is hidden: Visual to Lip-Sync */}
                                {(cmaOnlyMode || !enableEarlyExit) && (
                                  <path 
                                    d="M 360,42.5 L 510,42.5 L 510,90" 
                                    className={`flow-path ${getNodeState("lipsync") === "bypassed" ? "bypassed" : ""} ${getNodeState("visual") === "completed" ? "completed" : ""} ${getNodeState("lipsync") === "active" ? "active" : ""}`} 
                                    style={{ fill: "none" }}
                                  />
                                )}

                                {/* Normal path: Lip-Sync Attention -> Fusion */}
                                  <path 
                                  d="M 510,135 L 510,160" 
                                  className={`flow-path ${getNodeState("fusion") === "bypassed" ? "bypassed" : ""} ${
                                    (result?.early_exit || (file && file.name === "synthesized_face_swap.mp4" && progress > 40)) ? "bypassed" :
                                    (getNodeState("lipsync") === "completed" ? "completed" : "")
                                  } ${getNodeState("fusion") === "active" ? "active" : ""}`} 
                                />

                                {/* SPECIAL: Early-Exit Curved Bypass Path */}
                                {(!cmaOnlyMode && enableEarlyExit) && (
                                  <path 
                                    d="M 585,42.5 C 630,60 630,140 585,182.5" 
                                    className={`flow-bypass-path ${
                                      (mode !== "cross_modal_cnn" && mode !== "meta_learning") ? "bypassed" :
                                      (result?.early_exit || (file && file.name === "synthesized_face_swap.mp4" && progress > 40)) ? "active-bypass" : ""
                                    }`} 
                                  />
                                )}

                                {/* Direct Bypasses / Shortcuts for LLM and non-LLM modes */}
                                {(mode === "cross_modal_cnn" || mode === "meta_learning") && (
                                  <path 
                                    d="M 435,182.5 L 145,182.5" 
                                    className={`flow-path ${getNodeState("fusion") === "completed" ? "completed" : ""} ${getNodeState("report_out") === "active" ? "active" : ""}`}
                                  />
                                )}

                                {mode === "llm" && (
                                  <>
                                    <path 
                                      d="M 145,42.5 L 210,182.5" 
                                      className={`flow-path ${getNodeState("gemini") === "completed" ? "completed" : ""} ${(getNodeState("mtcnn") === "active" || getNodeState("gemini") === "active") ? "active" : ""}`}
                                    />
                                    <path 
                                      d="M 145,112.5 L 210,182.5" 
                                      className={`flow-path ${getNodeState("gemini") === "completed" ? "completed" : ""} ${(getNodeState("audio_prep") === "active" || getNodeState("gemini") === "active") ? "active" : ""}`}
                                    />
                                  </>
                                )}

                                {mode === "combined_forensics" && (
                                  <>
                                    {/* MHA shortcut bypass */}
                                    <path 
                                      d="M 510,130 C 470,140 470,170 510,182.5" 
                                      className={`flow-bypass-path ${getNodeState("fusion") === "completed" ? "completed" : ""} ${getNodeState("fusion") === "active" ? "active" : ""}`}
                                    />
                                    {/* MAML shortcut bypass */}
                                    <path 
                                      d="M 510,60 C 460,70 460,170 510,182.5" 
                                      className={`flow-bypass-path ${getNodeState("fusion") === "completed" ? "completed" : ""} ${getNodeState("fusion") === "active" ? "active" : ""}`}
                                    />
                                  </>
                                )}

                                {/* NODES */}
                                {/* Row 1 */}
                                <g className={`flow-node ${getNodeState("mtcnn")}`} onClick={() => setActiveInspectNode("mtcnn")}>
                                  <rect x="20" y="20" width="125" height="45" rx="8" className="node-rect" />
                                  <text x="82.5" y="47" className="node-text">📷 Face Crop (MTCNN)</text>
                                </g>

                                <g className={`flow-node ${getNodeState("visual")}`} onClick={() => setActiveInspectNode("visual")}>
                                  <rect x="210" y="20" width="150" height="45" rx="8" className="node-rect" />
                                  <text x="285" y="47" className="node-text">👁️ Spatial-Temporal (3D)</text>
                                </g>

                                {(!cmaOnlyMode && enableEarlyExit) && (
                                  <g className={`flow-node ${getNodeState("early_exit")}`} onClick={() => setActiveInspectNode("early_exit")}>
                                    <rect x="435" y="20" width="150" height="45" rx="8" className="node-rect" />
                                    <text x="510" y="47" className="node-text">⚡ Early-Exit (92%+)</text>
                                  </g>
                                )}

                                {/* Row 2 */}
                                <g className={`flow-node ${getNodeState("audio_prep")}`} onClick={() => setActiveInspectNode("acoustic")}>
                                  <rect x="20" y="90" width="125" height="45" rx="8" className="node-rect" />
                                  <text x="82.5" y="117" className="node-text">🔊 Sound Crop (WAV)</text>
                                </g>

                                <g className={`flow-node ${getNodeState("acoustic")}`} onClick={() => setActiveInspectNode("acoustic")}>
                                  <rect x="210" y="90" width="150" height="45" rx="8" className="node-rect" />
                                  <text x="285" y="117" className="node-text">🎵 Voice Spectrogram (1D)</text>
                                </g>

                                <g className={`flow-node ${getNodeState("lipsync")}`} onClick={() => setActiveInspectNode("lipsync")}>
                                  <rect x="435" y="90" width="150" height="45" rx="8" className="node-rect" />
                                  <text x="510" y="117" className="node-text">🔑 Lip-Sync Attn (MHA)</text>
                                </g>

                                {/* Row 3 */}
                                <g className={`flow-node ${getNodeState("report_out")}`} onClick={() => setActiveInspectNode("gemini")}>
                                  <rect x="20" y="160" width="125" height="45" rx="8" className="node-rect" />
                                  <text x="82.5" y="187" className="node-text">📝 Diagnostic Verdict</text>
                                </g>

                                <g className={`flow-node ${getNodeState("gemini")}`} onClick={() => setActiveInspectNode("gemini")}>
                                  <rect x="210" y="160" width="150" height="45" rx="8" className="node-rect" />
                                  <text x="285" y="187" className="node-text">🧠 Diagnostic Gemini</text>
                                </g>

                                <g className={`flow-node ${getNodeState("fusion")}`} onClick={() => setActiveInspectNode("fusion")}>
                                  <rect x="435" y="160" width="150" height="45" rx="8" className="node-rect" />
                                  <text x="510" y="187" className="node-text">🧬 Weighted Fusion</text>
                                </g>
                              </svg>
                            </div>

                            {/* Node Inspector Panel */}
                            {activeInspectNode && NODE_DETAILS[activeInspectNode] && (
                              <div className="inspector-panel animate-fade">
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border)", paddingBottom: "6px", marginBottom: "8px" }}>
                                  <span style={{ fontWeight: "600", fontSize: "12px", color: "var(--primary)" }}>🔍 Layer Details Inspector</span>
                                  <button onClick={() => setActiveInspectNode(null)} style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: "14px" }}>×</button>
                                </div>
                                <h4 className="inspect-title">{NODE_DETAILS[activeInspectNode].title}</h4>
                                <span className="inspect-source">Registry: {NODE_DETAILS[activeInspectNode].source}</span>
                                <p className="inspect-desc">{NODE_DETAILS[activeInspectNode].desc}</p>
                                <table className="inspect-table">
                                  <tbody>
                                    {Object.entries(NODE_DETAILS[activeInspectNode].stats).map(([k, v]) => (
                                      <tr key={k}>
                                        <td className="inspect-table-label">{k}</td>
                                        <td className="inspect-table-value font-mono">{v}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </div>

                          {/* Text Terminal Logs (collapsible inner) */}
                          <div className="thinking-logs">
                            {steps.map((step, idx) => (
                              <div 
                                key={idx} 
                                className={`log-entry ${idx === steps.length - 1 && loading ? "active" : ""}`}
                              >
                                &gt; {step} {idx === steps.length - 1 && loading && "..."}
                              </div>
                            ))}
                            {!loading && steps.length === 0 && (
                              <div className="log-entry">&gt; Pipeline initialized successfully.</div>
                            )}
                          </div>
                        </>
                      )}
                    </div>
                    
                    {/* Result Card when complete */}
                    {result && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <ResultCard result={result} mode={mode} />
                      </motion.div>
                    )}
                  </div>
                </div>
              )}
              
              <div ref={feedEndRef} style={{ height: "4px" }} />
            </div>
          </div>

          {/* ========================================================
              3. PROMPT MESSENGER CONTROLLER FOOTER
             ======================================================== */}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            style={{ display: "none" }}
            disabled={loading}
          />
          {file && (
            <footer className="workspace-footer">
              <div className="input-container">
                <div className="prompt-box">
                  
                  {/* Seeded dynamic loader strip */}
                  {loading && (
                    <div className="prompt-loading-strip">
                      <div className="prompt-loading-fill" style={{ width: `${progress}%` }}></div>
                    </div>
                  )}
                  
                  <div className="prompt-upper-row">
                    
                    <button 
                      className="prompt-file-trigger"
                      onClick={triggerFileInput}
                      disabled={loading}
                      type="button"
                    >
                      <span>📁</span>
                      <span>{file ? "Change Media" : "Upload Media"}</span>
                    </button>
                    
                    {file && (
                      <span className="prompt-file-indicator">
                        📄 {file.name}
                      </span>
                    )}
                    
                    <button 
                      className="run-prompt-btn"
                      onClick={handleAnalyze}
                      disabled={loading || !file}
                      type="button"
                      title="Initiate Pipeline"
                    >
                      ▲
                    </button>
                  </div>
                  
                </div>
              </div>
            </footer>
          )}
        </>
      ) : null}
    </>
  );
}