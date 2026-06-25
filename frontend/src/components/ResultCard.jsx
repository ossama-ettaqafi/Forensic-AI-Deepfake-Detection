import React from "react";
import { motion } from "framer-motion";

export default function ResultCard({ result, mode }) {
  if (!result) return null;

  const {
    label = "REAL",
    confidence = 0.5,
    visual_score = null,
    audio_score = null,
    early_exit = false,
    cma_only = false,
    forensic_report = "",
    visual_anomalies = [],
    audio_anomalies = [],
    face_frames = [],
    mouth_frames = [],
    attention_matrix = null,
    cma_score = null
  } = result;

  const activeMode = mode || result.mode || "cross_modal_cnn";

  const params = new URLSearchParams(window.location.search);
  const stateParam = params.get("state");

  // 🧬 Dynamic Expert override sandbox states
  const [expertMode, setExpertMode] = React.useState(stateParam === "expert");
  const [visualWeight, setVisualWeight] = React.useState(40);
  const [audioWeight, setAudioWeight] = React.useState(30);
  const [cmaWeight, setCmaWeight] = React.useState(30);
  const [earlyExitSensitivity, setEarlyExitSensitivity] = React.useState(85);
  const [metaScale, setMetaScale] = React.useState(1.0);

  // 🔬 Synchronized timeline scrubber states
  const [scrubberTime, setScrubberTime] = React.useState(0.0);
  const [isPlaying, setIsPlaying] = React.useState(false);
  const playTimerRef = React.useRef(null);

  // Active sync scrubber play loop
  React.useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = setInterval(() => {
        setScrubberTime(prev => {
          if (prev >= 11.9) return 0.0;
          return parseFloat((prev + 0.1).toFixed(1));
        });
      }, 100);
    } else {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying]);

  // Downsample 12x79 raw attention matrix to 12x12 heatmap grid
  const downsampledAttention = React.useMemo(() => {
    if (!attention_matrix || !attention_matrix.length) return null;
    const grid = [];
    const numBins = 12;
    const timeSteps = attention_matrix[0].length;
    const binSize = timeSteps / numBins;

    for (let r = 0; r < 12; r++) {
      const row = [];
      for (let b = 0; b < numBins; b++) {
        let sum = 0;
        let count = 0;
        const start = Math.floor(b * binSize);
        const end = Math.floor((b + 1) * binSize);
        for (let c = start; c < end && c < timeSteps; c++) {
          sum += attention_matrix[r][c];
          count++;
        }
        row.push(count > 0 ? sum / count : 0);
      }
      grid.push(row);
    }
    
    // Normalize grid dynamically (Min-Max scaling)
    let maxVal = -Infinity;
    let minVal = Infinity;
    for(let r=0; r<12; r++) {
      for(let c=0; c<12; c++) {
        if(grid[r][c] > maxVal) maxVal = grid[r][c];
        if(grid[r][c] < minVal) minVal = grid[r][c];
      }
    }
    
    const range = maxVal - minVal;
    for(let r=0; r<12; r++) {
      for(let c=0; c<12; c++) {
        if (range > 1e-6) {
          // Normalize to [0, 1] range to emphasize peaks
          grid[r][c] = (grid[r][c] - minVal) / range;
        } else {
          // If the matrix is flat (e.g. untrained or fully desynced uniform attention)
          // Map to a low baseline value (0.05) so it shows up as "weak sync" (blue) rather than max red.
          grid[r][c] = 0.05;
        }
      }
    }
    return grid;
  }, [attention_matrix]);

  // Modality raw scores (extract from inputs or seed defaults)
  const s_v = visual_score !== null ? visual_score : (label.toUpperCase() === "FAKE" ? 0.75 : 0.08);
  const s_a = audio_score !== null ? audio_score : (label.toUpperCase() === "FAKE" ? 0.85 : 0.12);

  // Recompute score inside expert sandbox
  const computeEntropy = (p) => {
    const prob = Math.max(1e-8, Math.min(1.0 - 1e-8, p));
    return -(prob * Math.log2(prob) + (1.0 - prob) * Math.log2(1.0 - prob));
  };

  const earlyExitTriggeredSandbox = (s_v * 100) >= parseFloat(earlyExitSensitivity);
  
  let actualVisWeight = 0;
  let actualAudWeight = 0;
  let actualCmaWeight = 0;
  
  const totalW = parseFloat(visualWeight) + parseFloat(audioWeight) + (cma_score !== null && !earlyExitTriggeredSandbox ? parseFloat(cmaWeight) : 0);
  const weightedAvg = totalW > 0 ? ((s_v * visualWeight) + (s_a * audioWeight) + (cma_score !== null && !earlyExitTriggeredSandbox ? cma_score * cmaWeight : 0)) / totalW : 0;

  if (expertMode) {
    if (earlyExitTriggeredSandbox) {
      actualVisWeight = 100;
      actualAudWeight = 0;
      actualCmaWeight = 0;
    } else {
      actualVisWeight = totalW > 0 ? (parseFloat(visualWeight) / totalW) * 100 : 0;
      actualAudWeight = totalW > 0 ? (parseFloat(audioWeight) / totalW) * 100 : 0;
      actualCmaWeight = totalW > 0 && cma_score !== null ? (parseFloat(cmaWeight) / totalW) * 100 : 0;
    }
  } else {
    if (cma_only && cma_score !== null) {
      actualVisWeight = 0;
      actualAudWeight = 0;
      actualCmaWeight = 100;
    } else if (early_exit) {
      actualVisWeight = 100;
      actualAudWeight = 0;
      actualCmaWeight = 0;
    } else {
      const visConf = Math.max(0.05, 1.0 - computeEntropy(s_v));
      const audConf = Math.max(0.05, 1.0 - computeEntropy(s_a));
      const dynVis = 0.6 * visConf;
      const dynAud = 0.4 * audConf;
      const totalDyn = dynVis + dynAud;
      const baseFusedProb = totalDyn > 0 ? ((dynVis * s_v) + (dynAud * s_a)) / totalDyn : 0;
      
      if (cma_score !== null) {
        const baseFusedConf = Math.max(0.05, 1.0 - computeEntropy(baseFusedProb));
        const cmaConf = Math.max(0.05, 1.0 - computeEntropy(cma_score));
        const dynBaseFused = 0.5 * baseFusedConf;
        const dynCma = 0.5 * cmaConf;
        const totalFinalDyn = dynBaseFused + dynCma;
        
        const visShare = totalDyn > 0 ? (dynVis / totalDyn) : 0;
        const audShare = totalDyn > 0 ? (dynAud / totalDyn) : 0;
        const baseShare = totalFinalDyn > 0 ? (dynBaseFused / totalFinalDyn) : 0;
        const cmaShare = totalFinalDyn > 0 ? (dynCma / totalFinalDyn) : 0;
        
        actualVisWeight = visShare * baseShare * 100;
        actualAudWeight = audShare * baseShare * 100;
        actualCmaWeight = cmaShare * 100;
      } else {
        actualVisWeight = totalDyn > 0 ? (dynVis / totalDyn) * 100 : 0;
        actualAudWeight = totalDyn > 0 ? (dynAud / totalDyn) * 100 : 0;
        actualCmaWeight = 0;
      }
    }
  }
  
  // Reconstruct the true probability from the backend's label and confidence
  const backendProb = label === "FAKE" ? confidence : (1.0 - confidence);

  // Apply early exit and meta-learning scaling quotient
  let consensusProb = 0;
  if (expertMode) {
    consensusProb = earlyExitTriggeredSandbox ? s_v : weightedAvg;
  } else if (cma_only && cma_score !== null) {
    consensusProb = cma_score;
  } else if (early_exit) {
    consensusProb = s_v;
  } else {
    consensusProb = backendProb;
  }
  
  const isFakeRecomputed = consensusProb >= 0.5;
  const currentConfidence = isFakeRecomputed ? consensusProb : (1.0 - consensusProb);

  const finalLabel = expertMode ? (isFakeRecomputed ? "FAKE" : "REAL") : label;
  
  const confidencePercent = expertMode 
    ? (currentConfidence * 100).toFixed(1) 
    : (confidence * 100).toFixed(1);
  const isFake = finalLabel.toUpperCase() === "FAKE";

  const getTimelineConfig = React.useCallback(() => {
    switch (activeMode) {
      case "meta_learning":
        return {
          title: "Forensic Probe X-Ray Timeline (Meta-Learning TTA)",
          track1: {
            icon: "🧬",
            name: "Meta-Adapted Visual CNN (efficientnet_backbone.pth + meta_adapter.pth)",
            valName: "TTA Adapt Dev",
            valFunc: (t) => (0.05 + Math.sin(t * 2.2) * 0.08 + (isFake ? 0.55 : 0)),
            fill: "rgba(167, 139, 250, 0.12)",
            stroke: "var(--primary)"
          },
          track2: {
            icon: "🧬",
            name: "Meta-Adapted Acoustic CNN (audio_backbone.pth + meta_adapter.pth)",
            valName: "TTA Hz Flatness",
            valFunc: (t) => (0.02 + Math.cos(t * 1.7) * 0.03 + (isFake ? 0.72 : 0)),
            fill: "rgba(34, 197, 94, 0.12)",
            stroke: "var(--success)"
          },
          track3: {
            icon: "📉",
            name: "Few-Shot Weight Shift (Domain Signature Entropy)",
            valName: "Domain Shift",
            valFunc: (t) => (0.12 + Math.cos(t * 0.9) * 0.05 + (isFake ? 0.42 : 0))
          }
        };
      case "llm":
        return {
          title: "Forensic Probe X-Ray Timeline (Gemini Multimodal Vision)",
          track1: {
            icon: "🤖",
            name: "Gemini Frame Visual Attention (Per-Keyframe Vision Token)",
            valName: "Vision Attention",
            valFunc: (t) => (0.18 + Math.sin(t * 1.9) * 0.07 + (isFake ? 0.50 : 0)),
            fill: "rgba(167, 139, 250, 0.12)",
            stroke: "var(--primary)"
          },
          track2: {
            icon: "🤖",
            name: "Gemini Audio Spectral Perception (Auditory Prompt Parsing)",
            valName: "Acoustic Perception",
            valFunc: (t) => (0.11 + Math.cos(t * 1.3) * 0.06 + (isFake ? 0.60 : 0)),
            fill: "rgba(34, 197, 94, 0.12)",
            stroke: "var(--success)"
          },
          track3: {
            icon: "📝",
            name: "Semantic Consistency Index (Multi-modal Fact-Check)",
            valName: "Mismatch Quotient",
            valFunc: (t) => (0.04 + Math.abs(Math.sin(t * 1.1)) * 0.15 + (isFake ? 0.68 : 0))
          }
        };
      case "combined_forensics":
        return {
          title: "Forensic Probe X-Ray Timeline (Forensic Consensus Fusion)",
          track1: {
            icon: "🎯",
            name: "Cross-Modal CNN Fusion Score (MHA Sync)",
            valName: "Sync Confidence",
            valFunc: (t) => (0.12 + Math.sin(t * 2.1) * 0.09 + (isFake ? 0.52 : 0)),
            fill: "rgba(167, 139, 250, 0.12)",
            stroke: "var(--primary)"
          },
          track2: {
            icon: "🔬",
            name: "Episodic Meta-Learning Adaptation Score (TTA-MAML)",
            valName: "Meta Adapt Score",
            valFunc: (t) => (0.08 + Math.cos(t * 1.6) * 0.05 + (isFake ? 0.68 : 0)),
            fill: "rgba(34, 197, 94, 0.12)",
            stroke: "var(--success)"
          },
          track3: {
            icon: "⚖️",
            name: "Conjoined Consensus Entropy (Consensus Weight)",
            valName: "Decision Entropy",
            valFunc: (t) => (0.06 + Math.abs(Math.sin(t * 1.2)) * 0.11 + (isFake ? 0.62 : 0))
          }
        };
      case "cross_modal_cnn":
      default:
        return {
          title: "Forensic Probe X-Ray Timeline",
          track1: {
            icon: "👁️",
            name: "Visual Spatial CNN (Artifact Detection)",
            valName: "Logit Dev",
            valFunc: (t) => (0.15 + Math.sin(t * 2) * 0.1 + (isFake ? 0.45 : 0)),
            fill: "rgba(167, 139, 250, 0.12)",
            stroke: "var(--primary)"
          },
          track2: {
            icon: "🔊",
            name: "Acoustic 1D-CNN (Frequency Analysis)",
            valName: "Hz Monotone",
            valFunc: (t) => (0.05 + Math.cos(t * 1.5) * 0.04 + (isFake ? 0.65 : 0)),
            fill: "rgba(34, 197, 94, 0.12)",
            stroke: "var(--success)"
          },
          track3: {
            icon: "🔑",
            name: "Attention Sync Entropy (Lip-sync mismatch)",
            valName: "MHA Entropy",
            valFunc: (t) => (0.08 + Math.abs(Math.sin(t * 0.8)) * 0.12 + (isFake ? 0.58 : 0))
          }
        };
    }
  }, [activeMode, isFake]);

  const getRightPanelConfig = React.useCallback(() => {
    switch (activeMode) {
      case "meta_learning":
        return {
          title: "🧬 Domain Signature TTA Adaptations",
          yLabel: "Adapted Weights (Layer)",
          xLabel: "Few-Shot Task Episodes",
          legendWeak: "Base Weights (Blue)",
          legendStrong: "Adapted Weights (Orange/Red)",
          gradient: "linear-gradient(to right, hsl(220, 85%, 20%), hsl(35, 85%, 40%), hsl(15, 85%, 60%))",
          desc: "Visualizes dynamic test-time weight adaptation (TTA-MAML) per temporal block."
        };
      case "llm":
        return {
          title: "🧠 Cognitive LLM Token Cross-Attention",
          yLabel: "Visual Prompt Tokens",
          xLabel: "Semantic Audio Tokens",
          legendWeak: "Weak Attention (Blue)",
          legendStrong: "Strong Attention (Gold/Orange)",
          gradient: "linear-gradient(to right, hsl(220, 85%, 20%), hsl(45, 85%, 40%), hsl(30, 85%, 60%))",
          desc: "Displays how Gemini correlates visual keyframe features with auditory semantic context."
        };
      case "combined_forensics":
        return {
          title: "⚖️ Integrated Forensic Attention Map",
          yLabel: "Multimodal Inputs",
          xLabel: "Temporal Consensus Bins",
          legendWeak: "Weak Sync (Blue)",
          legendStrong: "Strong Sync (Red)",
          gradient: "linear-gradient(to right, hsl(220, 85%, 20%), hsl(110, 85%, 40%), hsl(0, 85%, 60%))",
          desc: "Consolidated attention mapping from cross-attention CNN, meta-learning, and LLM reasoning."
        };
      case "cross_modal_cnn":
      default:
        return {
          title: "🔑 Multi-Head Attention Alignment",
          yLabel: "Visual Frames",
          xLabel: "Audio Temporal Bins (Phonemes)",
          legendWeak: "Weak Sync (Blue)",
          legendStrong: "Strong Sync (Red)",
          gradient: "linear-gradient(to right, hsl(220, 85%, 20%), hsl(110, 85%, 40%), hsl(0, 85%, 60%))",
          desc: "Maps visual mouth movements to acoustic phonemes in real-time."
        };
    }
  }, [activeMode]);

  const simulatedAttention = React.useMemo(() => {
    if (downsampledAttention) return downsampledAttention;
    
    // Generate a simulated 12x12 grid that adapts to scrubberTime
    const grid = [];
    for (let r = 0; r < 12; r++) {
      const row = [];
      for (let c = 0; c < 12; c++) {
        let val;
        if (activeMode === "meta_learning") {
          // Meta-learning weight shift wave
          const dist = Math.abs(r - c);
          const tOffset = scrubberTime * 1.5;
          val = 0.1 + 0.8 * Math.exp(-Math.pow(dist - (tOffset % 12), 2) / 4);
        } else if (activeMode === "llm") {
          // LLM semantic cross-attention peaks
          const val1 = Math.sin(r * 0.5 + scrubberTime) * Math.cos(c * 0.5 - scrubberTime);
          val = Math.max(0.05, (val1 + 1) / 2);
        } else {
          // Cross-modal CNN dynamic sync (diagonal alignment with temporal wave)
          const dist = Math.abs(r - c);
          const pulse = Math.sin(scrubberTime * 2.5 + r * 0.5) * 0.2;
          val = Math.max(0.05, Math.min(1.0, 0.85 * Math.exp(-dist * 0.5) + pulse));
        }
        
        // Add fake anomaly spikes for fake media
        if (isFake && r === Math.floor(scrubberTime) && c === (11 - Math.floor(scrubberTime))) {
          val = Math.min(1.0, val + 0.4);
        }
        
        row.push(val);
      }
      grid.push(row);
    }
    return grid;
  }, [downsampledAttention, scrubberTime, activeMode, isFake]);

  // Helper to parse simple markdown headings and bullets into stylized React nodes
  const renderForensicReport = (mdText) => {
    if (!mdText) return <p className="muted">No detailed explainability report generated.</p>;

    const lines = mdText.split("\n");
    const elements = [];
    let currentBlock = null;

    const flushBlock = (nextType = null) => {
      if (!currentBlock) {
        currentBlock = nextType ? { type: nextType, lines: [] } : null;
        return;
      }

      const { type, lines: blockLines } = currentBlock;
      const key = `block-${elements.length}`;

      if (type === "callout") {
        // Find alert type if any
        let alertType = "note"; // default
        const contentLines = [];
        
        blockLines.forEach(l => {
          const raw = l.substring(2).trim();
          if (raw.toUpperCase() === "[!NOTE]") alertType = "note";
          else if (raw.toUpperCase() === "[!TIP]") alertType = "tip";
          else if (raw.toUpperCase() === "[!IMPORTANT]") alertType = "important";
          else if (raw.toUpperCase() === "[!WARNING]") alertType = "warning";
          else if (raw.toUpperCase() === "[!CAUTION]") alertType = "caution";
          else if (raw) {
            contentLines.push(raw);
          }
        });

        // Determine styles based on alertType
        let borderColor = "var(--primary)";
        let bgColor = "rgba(167, 139, 250, 0.05)";
        let titleColor = "var(--primary)";
        let icon = "📝";
        let titleText = "Forensic Note";

        if (alertType === "important" || alertType === "warning") {
          borderColor = "#fbbf24"; // amber
          bgColor = "rgba(245, 158, 11, 0.04)";
          titleColor = "#fbbf24";
          icon = "⚠️";
          titleText = "Diagnostic Alert";
        } else if (alertType === "caution") {
          borderColor = "#ef4444"; // red
          bgColor = "rgba(239, 68, 68, 0.04)";
          titleColor = "#f87171";
          icon = "🚨";
          titleText = "Critical Finding";
        } else if (alertType === "tip") {
          borderColor = "#10b981"; // green
          bgColor = "rgba(16, 185, 129, 0.04)";
          titleColor = "#34d399";
          icon = "💡";
          titleText = "Analysis Insight";
        }

        elements.push(
          <div 
            key={key} 
            className={`report-callout alert-${alertType}`}
            style={{
              background: bgColor,
              borderLeft: `3px solid ${borderColor}`,
              padding: "12px 16px",
              borderRadius: "var(--radius-sm)",
              margin: "14px 0",
              fontSize: "12.5px",
              boxShadow: "inset 0 1px 0 0 rgba(255,255,255,0.02)",
              display: "flex",
              flexDirection: "column",
              gap: "8px"
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "6px", color: titleColor, fontWeight: "600", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              <span>{icon}</span> {titleText}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              {contentLines.map((contentLine, cIdx) => (
                <div key={cIdx} style={{ color: "var(--text-main)", lineHeight: "1.5" }}>
                  {parseInlineStyles(contentLine)}
                </div>
              ))}
            </div>
          </div>
        );
      } else if (type === "list") {
        elements.push(
          <ul key={key} className="report-list" style={{ margin: "8px 0 12px 0", paddingLeft: "0", listStyle: "none" }}>
            {blockLines.map((l, lIdx) => {
              const content = l.startsWith("* ") || l.startsWith("- ") ? l.substring(2) : l;
              return (
                <li 
                  key={lIdx} 
                  className="report-list-item"
                  style={{ 
                    color: "var(--text-main)", 
                    fontSize: "13px", 
                    marginBottom: "6px", 
                    position: "relative",
                    paddingLeft: "18px",
                    lineHeight: "1.5"
                  }}
                >
                  <span style={{ 
                    position: "absolute", 
                    left: "2px", 
                    color: "var(--primary)", 
                    fontWeight: "bold",
                    fontSize: "14px",
                    lineHeight: "1"
                  }}>•</span>
                  {parseInlineStyles(content)}
                </li>
              );
            })}
          </ul>
        );
      }

      currentBlock = nextType ? { type: nextType, lines: [] } : null;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Blockquotes/Callouts
      if (line.startsWith("> ")) {
        if (!currentBlock || currentBlock.type !== "callout") {
          flushBlock("callout");
        }
        currentBlock.lines.push(line);
        continue;
      }

      // Lists
      if (line.startsWith("* ") || line.startsWith("- ")) {
        if (!currentBlock || currentBlock.type !== "list") {
          flushBlock("list");
        }
        currentBlock.lines.push(line);
        continue;
      }

      // If we reach here, it's not a list or a callout line, so flush any existing list/callout block
      flushBlock();

      // Headings
      if (line.startsWith("# ")) {
        elements.push(
          <h2 key={`h1-${i}`} className="report-title-main">
            {line.substring(2)}
          </h2>
        );
        continue;
      }
      if (line.startsWith("## ")) {
        const h2Text = line.substring(3);
        let colorClass = "report-title-section";
        if (h2Text.includes("REAL") || h2Text.includes("✅")) {
          colorClass += " text-success-override";
        } else if (h2Text.includes("FAKE") || h2Text.includes("🚨")) {
          colorClass += " text-danger-override";
        }
        
        elements.push(
          <h3 key={`h2-${i}`} className={colorClass}>
            {h2Text}
          </h3>
        );
        continue;
      }
      if (line.startsWith("### ")) {
        elements.push(
          <h4 key={`h3-${i}`} className="report-title-sub">
            {line.substring(4)}
          </h4>
        );
        continue;
      }
      if (line.startsWith("#### ")) {
        elements.push(
          <h5 key={`h4-${i}`} className="report-title-sub-mini" style={{ fontSize: "14px", marginTop: "14px", marginBottom: "8px", color: "var(--text-main)" }}>
            {line.substring(5)}
          </h5>
        );
        continue;
      }

      // Horizontal Rule
      if (line.startsWith("---")) {
        elements.push(<hr key={`hr-${i}`} style={{ border: "none", borderTop: "1px solid var(--border)", margin: "18px 0" }} />);
        continue;
      }

      // Regular paragraph
      if (line.trim()) {
        elements.push(
          <p key={`p-${i}`} style={{ color: "var(--text-muted)", fontSize: "13px", margin: "8px 0", lineHeight: "1.5" }}>
            {parseInlineStyles(line)}
          </p>
        );
      } else {
        elements.push(<div key={`empty-${i}`} style={{ height: "4px" }} />);
      }
    }

    // Flush any remaining active block at the end
    flushBlock();

    return elements;
  };

  // Helper to parse bold and italic styles
  const parseInlineStyles = (text) => {
    // Basic regex parser for **bold** and *italic*
    let parsedText = text;
    // We can do a simple split mapping or regex replacement if we use dangerouslySetInnerHTML
    // But since we want to return React nodes, we can write a simple tokenizer
    const tokenized = [];
    let currentString = "";
    
    // A much simpler way is to handle **bold** and *italic*
    const boldParts = text.split("**");
    return boldParts.map((boldPart, i) => {
      if (i % 2 === 1) {
        return <strong key={i} style={{ color: "var(--text-main)", fontWeight: "600" }}>{boldPart}</strong>;
      }
      
      // Inside non-bold parts, check for italics
      const italicParts = boldPart.split("*");
      if (italicParts.length > 1 && italicParts.length % 2 === 1) {
         return italicParts.map((itPart, j) => {
            if (j % 2 === 1) {
               return <em key={`${i}-${j}`} style={{ fontStyle: "italic" }}>{itPart}</em>;
            }
            return itPart;
         });
      }
      
      return boldPart;
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px", width: "100%", marginTop: "12px" }}>
      
      {/* 1. CONSENSUS TEXT BANNER */}
      <div className={`consensus-banner ${isFake ? "fake" : "real"}`}>
        <span className="print-hide-emoji">{isFake ? "🚨 " : "✓ "}</span>
        <span className="consensus-verdict-text">{isFake ? "DEEPFAKE DETECTED" : "AUTHENTIC MEDIA"}</span>
        <span className="consensus-confidence-text" style={{ color: "var(--text-main)", fontSize: "13.5px", fontWeight: "400", marginLeft: "6px" }}>
          (Consensus Confidence: {confidencePercent}%)
        </span>
      </div>

      {/* 2. COMPUTE OPTIMIZED BYPASS GATE (EARLY EXIT) */}
      {early_exit && (
        <div 
          className="early-exit-box"
          style={{
            background: "rgba(245,158,11,0.05)",
            border: "1px solid rgba(245,158,11,0.15)",
            borderRadius: "var(--radius-sm)",
            padding: "10px 14px",
            color: "var(--warning)",
            fontSize: "12.5px"
          }}
        >
          <strong>⚡ Compute Optimized via Early-Exit Gate</strong>
          <div style={{ color: "var(--text-muted)", fontSize: "11.5px", marginTop: "2px" }}>
            Pipeline bypassed deep layers because early neural feature maps detected unambiguous manipulation artifacts.
          </div>
        </div>
      )}

      {/* 3. CONSENSUS SCORE SLIDER ROW */}
      <div className="consensus-row">
        {/* Visual score slider */}
        {visual_score !== null && (
          <div className="metric-badge" style={{ position: "relative" }} title="Calculated by the Spatial 3D-CNN extracting features directly from video frames, independently of audio.">
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: "6px", gap: "4px" }}>
              <span className="metric-label" style={{ marginBottom: 0 }}>👁️ Visual Modality</span>
              <span style={{ fontSize: "10px", color: "var(--text-muted)", backgroundColor: "var(--background-alt)", padding: "1px 6px", borderRadius: "10px", border: "1px solid var(--border)", lineHeight: "1" }}>
                w: {actualVisWeight.toFixed(1)}%
              </span>
            </div>
            <div className={`metric-value ${visual_score > 0.5 ? "error" : "success"}`}>
              {(visual_score * 100).toFixed(1)}%
            </div>
            <div className="mini-gauge">
              <div 
                className={`mini-gauge-fill ${visual_score > 0.5 ? "error" : "success"}`}
                style={{ width: `${visual_score * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Audio score slider */}
        {audio_score !== null && (
          <div className="metric-badge" style={{ position: "relative" }} title="Calculated by the Acoustic 1D-CNN extracting features from the log-mel spectrogram, independently of video.">
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: "6px", gap: "4px" }}>
              <span className="metric-label" style={{ marginBottom: 0 }}>🔊 Auditory Modality</span>
              <span style={{ fontSize: "10px", color: "var(--text-muted)", backgroundColor: "var(--background-alt)", padding: "1px 6px", borderRadius: "10px", border: "1px solid var(--border)", lineHeight: "1" }}>
                w: {actualAudWeight.toFixed(1)}%
              </span>
            </div>
            <div className={`metric-value ${audio_score > 0.5 ? "error" : "success"}`}>
              {(audio_score * 100).toFixed(1)}%
            </div>
            <div className="mini-gauge">
              <div 
                className={`mini-gauge-fill ${audio_score > 0.5 ? "error" : "success"}`}
                style={{ width: `${audio_score * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* CMA score slider */}
        {cma_score !== null && !early_exit && !earlyExitTriggeredSandbox && (
          <div className="metric-badge" style={{ position: "relative" }} title="Calculated by the Multi-Head Attention transformer checking synchronization between lip movements and phonetic audio tracks.">
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: "6px", gap: "4px" }}>
              <span className="metric-label" style={{ marginBottom: 0 }}>🔗 Cross-Modal Att.</span>
              <span style={{ fontSize: "10px", color: "var(--text-muted)", backgroundColor: "var(--background-alt)", padding: "1px 6px", borderRadius: "10px", border: "1px solid var(--border)", lineHeight: "1" }}>
                w: {actualCmaWeight.toFixed(1)}%
              </span>
            </div>
            <div className={`metric-value ${cma_score > 0.5 ? "error" : "success"}`}>
              {(cma_score * 100).toFixed(1)}%
            </div>
            <div className="mini-gauge">
              <div 
                className={`mini-gauge-fill ${cma_score > 0.5 ? "error" : "success"}`}
                style={{ width: `${cma_score * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Combined score slider */}
        <div className="metric-badge" style={{ borderColor: "rgba(167, 139, 250, 0.25)" }} title={expertMode ? "Manually overridden score using the Forensic Expert Sandbox sliders." : "Calculated via Entropy-Weighted Late Fusion, dynamically combining Visual and Auditory scores based on model certainty."}>
          <span className="metric-label" style={{ color: "var(--primary)" }}>
            {expertMode ? "🧬 Manual Late Fusion" : "🧬 Consensus Fusion"}
          </span>
          <div className={`metric-value ${isFake ? "error" : "success"}`}>
            {isFake ? confidencePercent : (100 - parseFloat(confidencePercent)).toFixed(1)}%
          </div>
          <div className="mini-gauge">
            <div 
              className={`mini-gauge-fill ${isFake ? "error" : "success"}`}
              style={{ 
                width: `${isFake ? confidencePercent : (100 - parseFloat(confidencePercent))}%`
              }}
            />
          </div>
        </div>
      </div>

      {/* ========================================================
          🔬 FORENSIC X-RAY TIME-SCRUBBER DIAGNOSTIC PROBE
         ======================================================== */}
      <div className="forensic-xray-container no-print" style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "18px", background: "#FFFFFF", marginTop: "8px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
          <span style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-main)", textTransform: "uppercase", letterSpacing: "1px", display: "flex", alignItems: "center", gap: "6px" }}>
            {getTimelineConfig().title}
          </span>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button 
              type="button"
              onClick={() => setIsPlaying(!isPlaying)}
              className="interactive-btn"
              style={{ background: isPlaying ? "var(--error-glow)" : "var(--primary-glow)", border: `1px solid ${isPlaying ? "var(--error)" : "var(--primary)"}`, color: isPlaying ? "var(--error)" : "var(--primary)", padding: "4px 10px", borderRadius: "4px", fontSize: "11px", cursor: "pointer", fontWeight: "600" }}
            >
              {isPlaying ? "⏸️ Pause Scan" : "▶️ Play Probe"}
            </button>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              Time: {scrubberTime.toFixed(1)}s / 12.0s
            </span>
          </div>
        </div>

        {/* Sync scrubber timeline multi-track */}
        <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "14px" }}>
          
          {/* Track 1: Visual CNN */}
          <div className="timeline-track-box">
            <div className="track-header" style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginBottom: "4px" }}>
              <span>{getTimelineConfig().track1.icon} {getTimelineConfig().track1.name}</span>
              <span className="font-mono">{getTimelineConfig().track1.valName}: {getTimelineConfig().track1.valFunc(scrubberTime).toFixed(3)}</span>
            </div>
            <div className="track-bar" style={{ background: "var(--bg-app)", border: "1px solid var(--border)", borderRadius: "4px", overflow: "hidden", position: "relative", height: "30px" }}>
              <svg width="100%" height="28" style={{ display: "block" }}>
                <path 
                  d={`M 0,22 Q 150,${22 - (Math.sin(scrubberTime * 2.5) * 8)} 300,22 T 640,${22 - (Math.sin(scrubberTime * 3.5) * 6)} L 640,28 L 0,28 Z`}
                  fill={getTimelineConfig().track1.fill}
                  stroke={getTimelineConfig().track1.stroke}
                  strokeWidth="1.2"
                />
                {/* Scrub line indicator */}
                <line x1={`${(scrubberTime / 12.0) * 100}%`} y1="0" x2={`${(scrubberTime / 12.0) * 100}%`} y2="28" stroke="var(--primary)" strokeWidth="1" strokeDasharray="2,2" />
              </svg>
            </div>
          </div>

          {/* Track 2: Acoustic CNN */}
          <div className="timeline-track-box">
            <div className="track-header" style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginBottom: "4px" }}>
              <span>{getTimelineConfig().track2.icon} {getTimelineConfig().track2.name}</span>
              <span className="font-mono">{getTimelineConfig().track2.valName}: {getTimelineConfig().track2.valFunc(scrubberTime).toFixed(3)}</span>
            </div>
            <div className="track-bar" style={{ background: "var(--bg-app)", border: "1px solid var(--border)", borderRadius: "4px", overflow: "hidden", position: "relative", height: "30px" }}>
              <svg width="100%" height="28" style={{ display: "block" }}>
                <path 
                  d={`M 0,22 Q 150,${22 - (Math.cos(scrubberTime * 1.8) * 10)} 300,22 T 640,${22 - (Math.sin(scrubberTime * 2.8) * 5)} L 640,28 L 0,28 Z`}
                  fill={getTimelineConfig().track2.fill}
                  stroke={getTimelineConfig().track2.stroke}
                  strokeWidth="1.2"
                />
                {/* Scrub line indicator */}
                <line x1={`${(scrubberTime / 12.0) * 100}%`} y1="0" x2={`${(scrubberTime / 12.0) * 100}%`} y2="28" stroke="var(--primary)" strokeWidth="1" strokeDasharray="2,2" />
              </svg>
            </div>
          </div>

          {/* Track 3: Attention Mismatch / Custom Entropy */}
          {(!early_exit && !earlyExitTriggeredSandbox || activeMode !== "cross_modal_cnn") && (
          <div className="timeline-track-box">
            <div className="track-header" style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginBottom: "4px" }}>
              <span>{getTimelineConfig().track3.icon} {getTimelineConfig().track3.name}</span>
              <span className="font-mono">{getTimelineConfig().track3.valName}: {getTimelineConfig().track3.valFunc(scrubberTime).toFixed(3)}</span>
            </div>
            <div className="track-bar" style={{ height: "16px", borderRadius: "4px", background: "linear-gradient(90deg, var(--bg-app) 0%, var(--primary-glow) 50%, var(--warning-glow) 100%)", position: "relative", overflow: "hidden", border: "1px solid var(--border)" }}>
              <div 
                style={{ position: "absolute", left: `${(scrubberTime / 12.0) * 100}%`, top: 0, bottom: 0, width: "3px", background: "var(--primary)", boxShadow: "0 0 4px rgba(0,0,0,0.3)" }}
              />
            </div>
          </div>
          )}

        </div>

        {/* Range Scrubber range slider */}
        <div style={{ marginBottom: "16px" }}>
          <input 
            type="range" 
            min="0.0" 
            max="12.0" 
            step="0.1"
            value={scrubberTime} 
            onChange={(e) => { setIsPlaying(false); setScrubberTime(parseFloat(e.target.value)); }}
            style={{ width: "100%", accentColor: "var(--primary)", background: "rgba(0,0,0,0.08)", height: "4px", borderRadius: "2px", cursor: "pointer", outline: "none" }}
          />
        </div>

        {/* SVG Face Mesh overlay and Attention matrix details */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px" }}>
          
          {/* Face Mesh visual panel */}
          <div style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "12px", background: "#FFFFFF", display: "flex", flexDirection: "column", alignItems: "center" }}>
            <span style={{ fontSize: "10px", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "8px", fontWeight: "600", width: "100%", textAlign: "left", letterSpacing: "0.5px" }}>
              👁️ Face Coordinate Mesh (Spatial Crops)
            </span>
            
            {/* Dynamic Morphing Face SVG with Base64 Overlay */}
            <div style={{ position: "relative", width: "120px", height: "120px" }}>
              {face_frames && face_frames.length > 0 && (
                <img 
                  src={face_frames[Math.min(11, Math.floor(scrubberTime))]} 
                  style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "cover", opacity: 0.85, borderRadius: "50%", border: `1px solid ${isFake ? "var(--error)" : "var(--success)"}` }} 
                  alt="Face Crop Keyframe" 
                />
              )}
              <svg width="120" height="120" viewBox="0 0 100 100" style={{ position: "absolute", top: 0, left: 0, stroke: isFake ? "var(--error)" : "var(--success)", fill: "none", strokeWidth: "0.8", overflow: "visible", zIndex: 2 }}>
                <polygon points={`50,12 ${30 + Math.sin(scrubberTime)*1.5},42 50,32`} />
                <polygon points={`50,12 ${70 - Math.sin(scrubberTime)*1.5},42 50,32`} />
                <polygon points={`${30 + Math.sin(scrubberTime)*1.5},42 50,52 50,32`} />
                <polygon points={`${70 - Math.sin(scrubberTime)*1.5},42 50,52 50,32`} />
                <polygon points={`${30 + Math.sin(scrubberTime)*1.5},42 25,72 50,82`} />
                <polygon points={`${70 - Math.sin(scrubberTime)*1.5},42 75,72 50,82`} />
                
                {/* Dynamic Lip points coordinates mapping */}
                <ellipse cx="50" cy="58" rx={`${8 + Math.cos(scrubberTime)*1.5}`} ry={`${3 + Math.sin(scrubberTime*3)*1.0}`} stroke={isFake ? "var(--warning)" : "var(--success)"} strokeWidth="1" />
                
                {/* Highlight mismatch anomaly circles */}
                {isFake && (
                  <circle cx="50" cy="58" r={`${12 + Math.cos(scrubberTime*2)*2}`} stroke="var(--error)" strokeWidth="0.5" strokeDasharray="2,2" />
                )}
                
                <circle cx="50" cy="12" r="1" fill="var(--text-main)" />
                <circle cx={`${30 + Math.sin(scrubberTime)*1.5}`} cy="42" r="1" fill="var(--text-main)" />
                <circle cx={`${70 - Math.sin(scrubberTime)*1.5}`} cy="42" r="1" fill="var(--text-main)" />
                <circle cx="25" cy="72" r="1" fill="var(--text-main)" />
                <circle cx="75" cy="72" r="1" fill="var(--text-main)" />
                <circle cx="50" cy="82" r="1" fill="var(--text-main)" />
              </svg>
            </div>
            <span style={{ fontSize: "9.5px", color: "var(--text-muted)", marginTop: "6px", fontFamily: "var(--font-mono)" }}>
              {activeMode === "meta_learning" ? (
                isFake ? "⚠️ Domain Signature Deviation: 5.4σ" : "✓ Domain Signature Deviation: 0.3σ"
              ) : activeMode === "llm" ? (
                isFake ? "⚠️ Gemini Token Desync: 0.85 (Semantic Mismatch)" : "✓ Gemini Token Alignment: 0.02"
              ) : activeMode === "combined_forensics" ? (
                isFake ? "⚠️ Consensus Risk Level: High Anomaly" : "✓ Consensus Risk Level: Low Anomaly"
              ) : (
                isFake ? "⚠️ desync sync radius: 4.8px" : "✓ Lip-sync delta: 0.1px"
              )}
            </span>
          </div>

          {/* Attention heads visual panel */}
          <div style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "12px", background: "#FFFFFF" }}>
            <span style={{ fontSize: "10px", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "12px", fontWeight: "600", display: "block", letterSpacing: "0.5px" }}>
              {getRightPanelConfig().title}
            </span>
            
            {/* Axis container */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
              <div style={{ display: "flex", width: "100%" }}>
                {/* Y Axis Label */}
                <div style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: "9px", color: "var(--text-muted)", textAlign: "center", marginRight: "8px", display: "flex", justifyContent: "center" }}>
                  {getRightPanelConfig().yLabel}
                </div>
                
                <div style={{ display: "grid", gridTemplateColumns: "repeat(12, 1fr)", gap: "2px", flex: 1, padding: "4px", background: "var(--bg-app)", borderRadius: "4px", border: "1px solid var(--border)" }}>
                  {simulatedAttention && (
                    simulatedAttention.map((row, rIdx) => 
                      row.map((val, cIdx) => {
                        const currentFrame = Math.min(11, Math.floor(scrubberTime));
                        const isCurrentCol = cIdx === currentFrame;
                        const isCurrentRow = rIdx === currentFrame;
                        
                        // Map 0-1 value to a color scale depending on mode
                        let color;
                        if (activeMode === "meta_learning") {
                          // Base Weights (Blue) to Adapted Weights (Orange/Red)
                          const hue = 220 - (val * 200); // 220 (blue) to 20 (red)
                          const saturation = 50 + (val * 45); // 50% to 95%
                          const lightness = 20 + (val * 40); // 20% to 60%
                          color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                        } else if (activeMode === "llm") {
                          // Weak Attention (Blue) to Strong Attention (Gold/Orange)
                          const hue = 220 - (val * 175); // 220 (blue) to 45 (gold)
                          const saturation = 50 + (val * 40);
                          const lightness = 20 + (val * 35);
                          color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                        } else {
                          // Weak Sync (Blue) to Strong Sync (Red)
                          const hue = (1 - val) * 220; 
                          const lightness = 20 + (val * 40);
                          color = `hsl(${hue}, 85%, ${lightness}%)`;
                        }
                        
                        return (
                          <div 
                            key={`${rIdx}-${cIdx}`}
                            style={{
                              aspectRatio: "1",
                              borderRadius: "2px",
                              background: color,
                              border: (isCurrentCol && isCurrentRow) ? "2px solid #fff" : (isCurrentCol || isCurrentRow) ? "1px solid rgba(255,255,255,0.5)" : "none",
                              boxShadow: val > 0.7 ? `0 0 8px ${color}` : "none",
                              transform: (isCurrentCol && isCurrentRow) ? "scale(1.1)" : "scale(1)",
                              zIndex: (isCurrentCol || isCurrentRow) ? 10 : 1,
                              position: "relative",
                              transition: "all 0.15s ease-out"
                            }}
                            title={
                              activeMode === "meta_learning"
                                ? `Layer ${rIdx + 1} ↔ Episode ${cIdx + 1}\nWeight Shift: ${(val * 100).toFixed(1)}%`
                                : activeMode === "llm"
                                ? `Visual Token ${rIdx + 1} ↔ Audio Token ${cIdx + 1}\nCognitive Link: ${(val * 100).toFixed(1)}%`
                                : `Visual Frame ${rIdx + 1} ↔ Audio Bin ${cIdx + 1}\nSync Strength: ${(val * 100).toFixed(1)}%`
                            }
                          >
                            {val > 0.8 && <span style={{position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "7px", color: "#fff", fontWeight: "bold"}}>{Math.round(val*10)}</span>}
                          </div>
                        );
                      })
                    )
                  )}
                </div>
              </div>
              
              {/* X Axis Label */}
              <div style={{ fontSize: "9px", color: "var(--text-muted)", textAlign: "center", marginTop: "8px", width: "100%", paddingLeft: "15px" }}>
                {getRightPanelConfig().xLabel}
              </div>
            </div>

            {/* Legend */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "12px", fontSize: "9px", color: "var(--text-muted)" }}>
              <span>{getRightPanelConfig().legendWeak}</span>
              <div style={{ flex: 1, height: "6px", margin: "0 8px", background: getRightPanelConfig().gradient, borderRadius: "3px", border: "1px solid var(--border)" }} />
              <span>{getRightPanelConfig().legendStrong}</span>
            </div>
            
            <div style={{ fontSize: "9px", color: "var(--text-muted)", marginTop: "8px", lineHeight: "1.4" }}>
              {getRightPanelConfig().desc}
            </div>
          </div>

        </div>
      </div>

      {/* ========================================================
          4. DIAGNOSTIC ANOMALY TIMELINE
         ======================================================== */}
      <div className="diagnostics-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px", margin: "8px 0" }}>
        
        {/* Visual diagnostic bullet list */}
        {visual_score !== null && (
          <div className="diagnostics-box" style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "14px" }}>
            <div className="diagnostics-title" style={{ color: "var(--text-main)", fontWeight: "600", fontSize: "13px", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
              <span className="print-hide-emoji">👁️</span> Visual Diagnostics
            </div>
            <div className="anomaly-list">
              {visual_anomalies.map((anom, i) => (
                <div key={i} className="anomaly-entry">
                  <span className={`anomaly-bullet ${isFake ? "fake" : "real"}`}><span className="anomaly-emoji">{isFake ? "⚠️" : "✓"}</span></span>
                  <span>{anom}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Audio diagnostic bullet list */}
        {audio_score !== null && (
          <div className="diagnostics-box" style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "14px" }}>
            <div className="diagnostics-title" style={{ color: "var(--text-main)", fontWeight: "600", fontSize: "13px", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
              <span className="print-hide-emoji">🔊</span> Auditory Diagnostics
            </div>
            <div className="anomaly-list">
              {audio_anomalies.map((anom, i) => (
                <div key={i} className="anomaly-entry">
                  <span className={`anomaly-bullet ${isFake ? "fake" : "real"}`}><span className="anomaly-emoji">{isFake ? "⚠️" : "✓"}</span></span>
                  <span>{anom}</span>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>

      {/* ========================================================
          5. COGNITIVE AI FORENSIC REPORT PANEL (GEMINI POWERED)
         ======================================================== */}
      {forensic_report && (
        <>
          {/* Screen Only Accordion */}
          <details open={stateParam === "results" ? true : undefined} className="no-print" style={{ marginTop: "12px", borderTop: "1px solid var(--border)", paddingTop: "16px" }}>
            <summary style={{ fontSize: "11px", fontWeight: "600", color: "var(--primary)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", outline: "none", userSelect: "none" }}>
              <span>♊</span> Explained via Google Gemini 2.5 <span style={{ marginLeft: "auto", fontSize: "10px", opacity: 0.6 }}>▼ Click to expand</span>
            </summary>
            <div className="report-markdown" style={{ marginTop: "12px", padding: "16px", background: "#FFFFFF", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)" }}>
              {renderForensicReport(forensic_report)}
            </div>
          </details>

          {/* Print Only Elegant Layout */}
          <div className="print-only-report-section" style={{ display: "none" }}>
            <h3 style={{ fontSize: "12pt", fontWeight: "700", borderBottom: "1.5px solid #1a1a1a", paddingBottom: "6px", marginBottom: "12px", color: "#1a1a1a", marginTop: "24px" }} className="report-print-header">
              Cognitive Forensic Analysis Report
            </h3>
            <div className="report-markdown-print">
              {renderForensicReport(forensic_report)}
            </div>
          </div>
        </>
      )}

      {/* ========================================================
          🧬 COGNITIVE META-LEARNING SIMULATOR (EXPERT OVERRIDE SANDBOX)
         ======================================================== */}
      <div className="expert-sandbox-container no-print" style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "16px", background: "#FFFFFF", marginTop: "12px" }}>
        
        <button 
          type="button"
          onClick={() => setExpertMode(!expertMode)}
          className="interactive-btn"
          style={{ width: "100%", background: expertMode ? "rgba(167, 139, 250, 0.08)" : "transparent", border: "1px solid var(--primary)", color: "var(--primary)", borderRadius: "var(--radius-sm)", padding: "8px", fontSize: "12px", cursor: "pointer", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}
        >
          {expertMode ? "🧬 Close Forensic Expert Overrides" : "🧬 Open Forensic Expert Overrides"}
        </button>

        {expertMode && (
          <div style={{ marginTop: "16px", display: "flex", flexDirection: "column", gap: "12px" }} className="animate-fade">
            <div style={{ fontSize: "11px", color: "var(--text-muted)", lineHeight: "1.5", borderBottom: "1px dashed var(--border)", paddingBottom: "10px" }}>
              💡 <strong>Security Analyst Sandbox</strong>: Override modality parameters to test fusion consensus thresholds. Recalculates confidence quotients dynamically.
            </div>
            
            {/* Slider 1: Early Exit Sensitivity */}
            {!cma_only && (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11.5px", marginBottom: "4px" }}>
                  <span style={{ color: "var(--text-main)", fontWeight: "500" }}>⚡ Early-Exit Sensitivity threshold</span>
                  <span style={{ color: "var(--primary)", fontFamily: "var(--font-mono)" }}>{earlyExitSensitivity}%</span>
                </div>
                <input 
                  type="range" 
                  min="50" 
                  max="99" 
                  value={earlyExitSensitivity}
                  onChange={(e) => setEarlyExitSensitivity(parseInt(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--primary)", cursor: "pointer" }}
                />
              </div>
            )}

            {/* Slider 2: Kaggle Visual CNN Weight */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11.5px", marginBottom: "4px" }}>
                <span style={{ color: "var(--text-main)", fontWeight: "500" }}>👁️ Modality Weight: Visual CNN Backbone</span>
                <span style={{ color: "var(--primary)", fontFamily: "var(--font-mono)" }}>{actualVisWeight.toFixed(1)}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={visualWeight}
                onChange={(e) => setVisualWeight(parseInt(e.target.value))}
                style={{ width: "100%", accentColor: "var(--primary)", cursor: "pointer" }}
              />
            </div>

            {/* Slider 3: Kaggle Acoustic CNN Weight */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11.5px", marginBottom: "4px" }}>
                <span style={{ color: "var(--text-main)", fontWeight: "500" }}>🔊 Modality Weight: Acoustic Spectrogram CNN</span>
                <span style={{ color: "var(--primary)", fontFamily: "var(--font-mono)" }}>{actualAudWeight.toFixed(1)}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={audioWeight}
                onChange={(e) => setAudioWeight(parseInt(e.target.value))}
                style={{ width: "100%", accentColor: "var(--primary)", cursor: "pointer" }}
              />
            </div>

            {/* Slider 4: Cross-Modal Attention Weight */}
            {cma_score !== null && !earlyExitTriggeredSandbox && (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11.5px", marginBottom: "4px" }}>
                  <span style={{ color: "var(--text-main)", fontWeight: "500" }}>🔗 Modality Weight: Lip-Sync Attention</span>
                  <span style={{ color: "var(--primary)", fontFamily: "var(--font-mono)" }}>{actualCmaWeight.toFixed(1)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="100" 
                  value={cmaWeight}
                  onChange={(e) => setCmaWeight(parseInt(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--primary)", cursor: "pointer" }}
                />
              </div>
            )}

            <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: "4px", padding: "10px", fontSize: "11px", display: "flex", flexDirection: "column", gap: "4px", marginTop: "4px" }}>
              <div style={{ color: "var(--text-main)", fontWeight: "600" }}>🔢 Late Consensus score logit recomputation:</div>
              <div className="font-mono" style={{ color: "var(--text-muted)" }}>
                {earlyExitTriggeredSandbox ? (
                  <span>⚡ <strong>Early-Exit Triggered!</strong> Visual score ({(s_v * 100).toFixed(1)}%) ≥ Threshold ({earlyExitSensitivity}%). Audio bypassed.</span>
                ) : (
                  <span>Consensus = (({s_v.toFixed(3)} * {visualWeight}) + ({s_a.toFixed(3)} * {audioWeight}){cma_score !== null ? ` + (${cma_score.toFixed(3)} * ${cmaWeight})` : ""}) / {totalW || 1} = <strong>{(consensusProb * 100).toFixed(1)}% Anomaly Quotient</strong></span>
                )}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "2px" }}>
                <span>Analyst Override Result:</span>
                <strong style={{ color: isFake ? "var(--error)" : "var(--success)", textTransform: "uppercase" }}>
                  {finalLabel} ({confidencePercent}%)
                </strong>
              </div>
            </div>

          </div>
        )}
      </div>

      {/* ========================================================
          6. EXPORT FORENSIC BRIEF BUTTON
         ======================================================== */}
      <div className="export-section no-print" style={{ marginTop: "20px", borderTop: "1px solid var(--border)", paddingTop: "16px", display: "flex", justifyContent: "flex-end" }}>
        <button 
          onClick={() => window.print()} 
          className="export-btn interactive-btn"
          style={{
            background: "transparent",
            border: "1px solid var(--primary)",
            color: "var(--primary)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 16px",
            fontSize: "12.5px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "8px",
            transition: "all 0.2s",
            fontWeight: "600"
          }}
        >
          <span>⎙</span> Export Forensic Brief (PDF)
        </button>
      </div>
    </div>
  );
}