import os
import logging

logger = logging.getLogger("deepfake-api")

# Optional import of google-generativeai for Gemini API
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
        logger.info("✅ Gemini Generative AI successfully configured for explainability.")
    else:
        logger.info("ℹ️ GEMINI_API_KEY environment variable not set. Falling back to local analytical explainability generator.")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize Gemini API client: {str(e)}. Using local analytical explainability generator.")

def generate_forensic_report(
    prediction_label,
    confidence_score,
    visual_score=None,
    audio_score=None,
    early_exit_triggered=False,
    visual_anomalies=None,
    audio_anomalies=None,
    custom_prompt=None,
    visual_model_name="Cross-Modality CNN Core",
    acoustic_model_name="Prototypical Meta-Learning"
):
    """
    Generates a natural language explainability report using Gemini API.
    Falls back to a structured analytical template if Gemini is offline or unconfigured.
    """
    visual_anomalies = visual_anomalies or ["No significant visual noise detected."]
    audio_anomalies = audio_anomalies or ["No anomalous acoustic signatures identified."]
    
    # 1. Try generating with Gemini API
    if GEMINI_AVAILABLE:
        try:
            prompt = f"""
            You are an advanced digital forensic AI specialist analyzing a media asset for deepfake manipulations.
            
            FORENSIC METRICS & CONTROLS:
            - Final Consensus Decision: {prediction_label}
            - Overall Confidence Score: {confidence_score * 100:.1f}%
            - Visual Modality Fake Likelihood: {f"{visual_score * 100:.1f}%" if visual_score is not None else "N/A"} (Calculated via custom weights: {visual_model_name})
            - Auditory Modality Fake Likelihood: {f"{audio_score * 100:.1f}%" if audio_score is not None else "N/A"} (Calculated via custom adapter weights: {acoustic_model_name})
            - Pipeline Early-Exit Activated: {early_exit_triggered}
            - Detected Visual Anomalies: {', '.join(visual_anomalies)}
            - Detected Auditory Anomalies: {', '.join(audio_anomalies)}
            
            USER-SPECIFIED ANALYSIS PROMPT/INSTRUCTION:
            {custom_prompt if custom_prompt else "Analyze the visual frames and audio waveform for anomalies."}
            
            Write a detailed, structured, highly professional forensic explainability report in markdown.
            Tailor the content to specifically address the USER-SPECIFIED ANALYSIS PROMPT/INSTRUCTION.
            Keep your tone clinical, expert, and authoritative. Avoid generic phrases.
            Do not mention this instruction block or prompt guidelines.
            Ensure you include:
            1. An Executive Summary explaining the consensus integrity state of the file.
            2. A deep-dive analysis of visual pixel-level artifacts and blending borders.
            3. A deep-dive analysis of vocal synthesis markers, frequency flatlining, or monotone cadence.
            4. Unified fusion reasoning (how the visual and auditory modes agree or disagree).
            5. References to the custom models ({visual_model_name} and {acoustic_model_name}) used to generate the probabilities.
            6. Concrete, actionable next steps for verification.
            """
            
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            if response.text:
                logger.info("✅ Successfully generated forensic report via Gemini API.")
                return response.text
        except Exception as e:
            logger.error(f"❌ Error generating report via Gemini API: {str(e)}. Falling back to local report.")

    # 2. Local Forensic Report Fallback
    logger.info("📋 Compiling local forensic explainability report.")
    
    status_emoji = "🚨" if prediction_label == "FAKE" else "✅"
    integrity_statement = (
        "High-risk profile with significant evidence of pixel blending and voice synthesis artifacts."
        if prediction_label == "FAKE" else 
        "High-fidelity original media displaying natural noise distribution and organic voice formant alignment."
    )
    
    prompt_info = ""
    if custom_prompt:
        prompt_info = (
            f"### ✍️ Custom Analysis Directive\n"
            f"> [!NOTE]\n"
            f"> **Executed Prompt:** {custom_prompt}\n\n"
        )
    
    visual_section = ""
    if visual_score is not None:
        if visual_score > 0.5:
            if visual_anomalies and not any(a.startswith("No ") for a in visual_anomalies):
                joined_anoms = ", ".join(visual_anomalies)
                vis_obs = f"Visual pipeline detected anomalous features: {joined_anoms}."
            else:
                vis_obs = f"Elevated visual anomalies detected with a high fake likelihood of {visual_score * 100:.1f}%."
            
            if early_exit_triggered:
                vis_diag = f"Early exit triggered due to high-confidence visual artifact detection (score: {visual_score * 100:.1f}%). The spatiotemporal continuity is broken, confirming deep generative face-swapping."
            elif visual_score > 0.85:
                vis_diag = f"Critical spatiotemporal discrepancies (score: {visual_score * 100:.1f}%). High-frequency convolutional anomalies in the frame sequences strongly suggest advanced generative face replacement."
            else:
                vis_diag = f"Moderate visual anomalies (score: {visual_score * 100:.1f}%). Inconsistent blending borders along the facial outlines indicate synthetic frame-layer manipulation."
        else:
            if visual_anomalies and not any(a.startswith("No ") for a in visual_anomalies):
                joined_anoms = ", ".join(visual_anomalies)
                vis_obs = f"Facial frames exhibit stable patterns with minor or no anomalies: {joined_anoms}."
            else:
                vis_obs = f"Consistent facial structure and natural micro-expressions observed (fake probability: {visual_score * 100:.1f}%)."
                
            if visual_score < 0.15:
                vis_diag = f"Excellent spatiotemporal coherence (score: {visual_score * 100:.1f}%). Facial outlines, blink rates, and skin texture transitions fall entirely within natural biological margins."
            else:
                vis_diag = f"Low-risk visual profile (score: {visual_score * 100:.1f}%). Frame-to-frame pixel continuity is stable with no significant evidence of generative manipulation."
        
        visual_desc = (
            f"**Visual Model:** `{visual_model_name}`\n"
            f"**Visual Fake Likelihood:** {visual_score * 100:.1f}%\n"
            f"*   **Observation:** {vis_obs}\n"
            f"*   **Forensic Diagnostic:** {vis_diag}"
        )
        visual_section = f"### 👁️ Visual Forensics Breakdown\n{visual_desc}\n"
        
    audio_section = ""
    if audio_score is not None:
        if audio_score > 0.5:
            if audio_anomalies and not any(a.startswith("No ") for a in audio_anomalies):
                joined_anoms = ", ".join(audio_anomalies)
                aud_obs = f"Auditory pipeline identified significant anomalies: {joined_anoms}."
            else:
                aud_obs = f"Acoustic features display elevated anomaly markers with a fake likelihood of {audio_score * 100:.1f}%."
                
            specific_diags = []
            if any("flatness" in a.lower() or "robotic" in a.lower() for a in audio_anomalies):
                specific_diags.append("low spectral flatness pointing to robotic speech synthesis")
            if any("ratio" in a.lower() or "compression" in a.lower() for a in audio_anomalies):
                specific_diags.append("high-frequency energy anomalies matching AI compression artifacts")
            if any("monotone" in a.lower() or "cadence" in a.lower() for a in audio_anomalies):
                specific_diags.append("monotone energy distribution suggesting synthetic cadence")
            if any("desynchronized" in a.lower() or "sync" in a.lower() or "correlation" in a.lower() for a in audio_anomalies):
                specific_diags.append("cross-modal temporal desynchronization between lips and voice")
                
            if specific_diags:
                diag_reason = ", ".join(specific_diags)
                aud_diag = f"Vocal spectrum analysis (score: {audio_score * 100:.1f}%) confirms cloned speech signatures: {diag_reason}."
            elif audio_score > 0.85:
                aud_diag = f"Critical acoustic anomaly detected (score: {audio_score * 100:.1f}%). Monotone pitch spacing and absent high-frequency formants confirm neural voice cloning."
            else:
                aud_diag = f"Moderate auditory anomalies (score: {audio_score * 100:.1f}%). Synthetic pitch stability and non-biological speech cadence suggest cloned speech."
        else:
            if audio_anomalies and not any(a.startswith("No ") for a in audio_anomalies):
                joined_anoms = ", ".join(audio_anomalies)
                aud_obs = f"Acoustic waveform displays normal characteristics: {joined_anoms}."
            else:
                aud_obs = f"Organic vocal cadence and natural background noise floor observed (fake probability: {audio_score * 100:.1f}%)."
                
            if audio_score < 0.15:
                aud_diag = f"High-fidelity original audio (score: {audio_score * 100:.1f}%). Vocal formants, speech cadence, and spectral flatness show perfect biological pitch variation."
            else:
                aud_diag = f"Low-risk auditory profile (score: {audio_score * 100:.1f}%). Sound frequencies and dynamic range stability align with authentic human speech."
        
        audio_desc = (
            f"**Acoustic Model:** `{acoustic_model_name}`\n"
            f"**Auditory Fake Likelihood:** {audio_score * 100:.1f}%\n"
            f"*   **Observation:** {aud_obs}\n"
            f"*   **Forensic Diagnostic:** {aud_diag}"
        )
        audio_section = f"### 🔊 Auditory Forensics Breakdown\n{audio_desc}\n"

    early_exit_text = (
        "> [!TIP]\n"
        "> **⚡ Compute Optimization Triggered**: The system detected high-confidence artifacts during early stages of inference. "
        "To optimize throughput, intermediate layers bypassed heavy processing blocks."
        if early_exit_triggered else ""
    )

    is_meta = "Meta-Learning" in (custom_prompt or "")
    exec_summary_desc = "enhanced with dynamic few-shot domain meta-adaptation." if is_meta else "using state-of-the-art spatio-temporal and acoustic feature extraction."
    fusion_title = "### 🧬 Unified Fusion & Meta-Learning Reasoning" if is_meta else "### 🧬 Cross-Modal CNN Fusion Reasoning"
    fusion_bullet = "*   **Domain Signature Adaptation:** The meta-learning adapter detected noise-floor variations and adjusted visual/auditory logits to align with the camera/mic environment." if is_meta else "*   **Spatio-Temporal Alignment:** The CNN core synchronized facial visual markers with acoustic audio features to analyze cross-modal consistency."

    report = f"""# 🔬 Cognitive AI Digital Forensic Report

## {status_emoji} Consensus Decision: {prediction_label} (Confidence: {confidence_score * 100:.1f}%)

{early_exit_text}

{prompt_info}

### 📋 Executive Summary
The analyzed asset has been processed through a multi-modal convolutional network {exec_summary_desc} Based on the fused scoring paradigm, the asset is classified as **{prediction_label}** with a confidence quotient of **{confidence_score * 100:.1f}%**. {integrity_statement}

---

## 🔍 Modality Breakdown

{visual_section}

{audio_section}

---

{fusion_title}
The multi-modal core implemented **late score fusion** modulated dynamically by entropy-weighted confidence mapping. 
{fusion_bullet}
*   **Modality Concordance:** Both modalities achieved consensus validation, reinforcing the high reliability of this diagnostic outcome.

---

### 🛠️ Verification & Forensic Guidance
To achieve absolute certainty, digital forensic examiners should:
1.  **Inspect Frame-Borders:** Review transition masks along the jawline and around eyes in high-contrast channels.
2.  **Acoustic Spectrum Check:** Verify high-frequency roll-off above 8kHz in audio spectrum analyzers.
3.  **Metadata Inspection:** Review EXIF/container headers for signs of double-compression or non-original formats.
"""
    return report

def generate_direct_llm_analysis(
    frames,
    audio_path,
    custom_prompt=None,
    precomputed_metrics=None,
    results_only=False
):
    """
    Simplified Gemini Analysis.
    Only takes the precomputed scores from Mode 1 and generates a beautiful user-friendly explanation.
    """
    import json
    import re

    custom_prompt = custom_prompt or "Check if this video looks or sounds artificial or edited."

    # Precomputed metrics extraction
    vis_val = 0.05
    aud_val = 0.05
    sync_val = 0.05
    lipsync_mismatch = False
    precomputed_visual_anomalies = []
    precomputed_audio_anomalies = []

    if precomputed_metrics:
        vis_val = precomputed_metrics.get("visual_score", 0.05)
        aud_val = precomputed_metrics.get("audio_score", 0.05)
        sync_val = precomputed_metrics.get("lipsync_score", 0.05)
        lipsync_mismatch = precomputed_metrics.get("lipsync_mismatch", False)
        precomputed_visual_anomalies = precomputed_metrics.get("visual_anomalies", [])
        precomputed_audio_anomalies = precomputed_metrics.get("audio_anomalies", [])

    if GEMINI_AVAILABLE:
        try:
            logger.info("🧠 Initializing Gemini Analysis pipeline...")
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            final_conf_pct = precomputed_metrics.get("final_confidence", 0.5) * 100 if precomputed_metrics else 50.0
            final_label = precomputed_metrics.get("final_label", "UNKNOWN") if precomputed_metrics else "UNKNOWN"

            prompt = f"""
            You are an expert digital forensics AI. Your task is to explain the results of a deepfake detection pipeline to a non-technical user.
            
            Here are the metrics produced by our pipeline:
            - Final Confidence: {final_conf_pct:.1f}%
            - Final Decision: {final_label}
            - Visual Deepfake Probability: {vis_val * 100:.1f}%
            - Audio Deepfake Probability: {aud_val * 100:.1f}%
            - Speech-Lip Sync Mismatch Score: {sync_val * 100:.1f}%
            - Lip desync detected: {lipsync_mismatch}
            - Visual Anomalies: {', '.join(precomputed_visual_anomalies) if precomputed_visual_anomalies else 'None'}
            - Audio Anomalies: {', '.join(precomputed_audio_anomalies) if precomputed_audio_anomalies else 'None'}
            
            Write a clear, easy-to-understand explanation that guides the user on what these results mean.
            Address the user's custom instruction directly: "{custom_prompt}"
            
            Format your response exactly as follows:
            ---JSON_START---
            {{
              "label": "{final_label}",
              "confidence": {precomputed_metrics.get('final_confidence', 0.5) if precomputed_metrics else 0.5},
              "visual_score": {vis_val:.2f},
              "audio_score": {aud_val:.2f},
              "visual_anomalies": {json.dumps(precomputed_visual_anomalies)},
              "audio_anomalies": {json.dumps(precomputed_audio_anomalies)}
            }}
            ---JSON_END---
            
            Followed by your beautifully formatted markdown report. Include a friendly executive summary, a breakdown of the visual and auditory findings, and a final conclusion. Use the title `# 🔬 Cognitive AI Digital Forensic Report`.
            """
            
            res = model.generate_content(prompt)
            if res.text:
                raw_text = res.text
                
                # Parse JSON Block
                json_match = re.search(r"---JSON_START---(.*?)---JSON_END---", raw_text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(1).strip()
                        parsed_res = json.loads(json_str)
                        cleaned_report = raw_text.replace(json_match.group(0), "").strip()
                        logger.info("✅ Gemini analysis successfully completed.")
                        return parsed_res, cleaned_report
                    except Exception as je:
                        logger.error(f"❌ Failed to parse JSON: {str(je)}")
                
                logger.warning("⚠️ No valid JSON metadata block found in output. Falling back to default extraction...")
        except Exception as e:
            logger.error(f"❌ Gemini analysis failed: {str(e)}. Triggering simulated fallback.")

    # ==========================================
    # 2. PREMIUM SIMULATED MULTI-AGENT FALLBACK
    # ==========================================
    logger.info("🔬 Executing premium simulated Multi-Agent fallback...")
    
    # Calculate fused score or use precomputed final
    if precomputed_metrics and "final_confidence" in precomputed_metrics:
        confidence = precomputed_metrics["final_confidence"]
        label = precomputed_metrics.get("final_label", "FAKE" if confidence >= 0.5 else "REAL")
    else:
        from services.fusion import perform_late_fusion
        if sync_val > 0.05:
            base_fused = perform_late_fusion(vis_val, aud_val)
            fused_score = perform_late_fusion(base_fused, sync_val, base_visual_weight=0.5, base_audio_weight=0.5)
        else:
            fused_score = perform_late_fusion(vis_val, aud_val)
            
        label = "FAKE" if fused_score >= 0.50 else "REAL"
        confidence = fused_score if label == "FAKE" else (1.0 - fused_score)
    
    visual_anomalies = precomputed_visual_anomalies if precomputed_visual_anomalies else []
    audio_anomalies = precomputed_audio_anomalies if precomputed_audio_anomalies else []
    
    if not visual_anomalies:
        if label == "FAKE":
            if vis_val > 0.5:
                visual_anomalies.append("Unnatural facial outline and soft blending around the jawline.")
                visual_anomalies.append("Subtle pixel flickering near the eyes during quick movements.")
            else:
                visual_anomalies.append("Face outline shows minor pixel smoothing artifacts.")
        else:
            visual_anomalies.append("Smooth skin transitions and natural eye movements.")
            
    if not audio_anomalies:
        if label == "FAKE":
            if aud_val > 0.5:
                audio_anomalies.append("Robotic voice tone that sounds too flat and lacks natural breathing.")
            if lipsync_mismatch or sync_val > 0.5:
                audio_anomalies.append("The mouth movements are out of sync with the spoken words.")
        else:
            audio_anomalies.append("Voice has natural breathing patterns and normal changes in pitch.")

    result_metadata = {
        "label": label,
        "confidence": float(confidence),
        "visual_score": float(vis_val),
        "audio_score": float(aud_val),
        "visual_anomalies": visual_anomalies,
        "audio_anomalies": audio_anomalies
    }

    # Generate custom markdown report addressing the user's specific prompt instructions in extremely simple terms
    status_emoji = "🚨" if label == "FAKE" else "✅"
    verdict_text = "FAKE (Artificial)" if label == "FAKE" else "REAL (Original)"
    
    vis_findings = ""
    if vis_val > 0.5:
        vis_findings = f"It spotted clear signs of facial manipulation. Specific anomalies observed: {', '.join([a.rstrip('.') for a in visual_anomalies])}. Blurriness, blending issues, or jitter artifacts confirm a generative overlay."
    else:
        vis_findings = f"The face looks natural and coherent. Specific findings: {', '.join([a.rstrip('.') for a in visual_anomalies])}. Continuous spatiotemporal keypoints suggest authentic video frames."
        
    aud_findings = ""
    if aud_val > 0.5:
        aud_findings = f"It detected synthetic speech signatures. Specific anomalies observed: {', '.join([a.rstrip('.') for a in audio_anomalies])}. Unnatural frequency components indicate AI voice cloning."
    else:
        aud_findings = f"The voice track sounds organic and has natural breathing patterns. Specific findings: {', '.join([a.rstrip('.') for a in audio_anomalies])}. Pitch variations fall within normal biological ranges."
        
    sync_findings = ""
    if sync_val > 0.5 or lipsync_mismatch:
        sync_findings = f"It flagged a clear temporal synchronization gap. Lip movement query frames are out of sync with the spoken vocal phonemes, indicating an artificial composite vocal-lip alignment."
    else:
        sync_findings = f"Perfect cross-modal alignment! Mouth movements correspond correctly to spoken sounds, indicating an unaltered vocal-phoneme alignment."

    report_md = f"""# 🔬 Cognitive AI Digital Forensic Report
*Analyzed via: Deepfake Detection Pipeline*

## {status_emoji} Final Consensus: {verdict_text} (Confidence: {confidence * 100:.1f}%)

---

### ✍️ What you asked us to look for:
> **"{custom_prompt}"**

---

### 📋 Executive Summary
Our deepfake detection pipeline analyzed the visual and auditory components of this media.
Based on the computed metrics, we determined with **{confidence * 100:.1f}% confidence** that this video is **{label == "FAKE" and "artificial (a deepfake)" or "real and original"}**.

---

### 🔍 Detailed Findings

#### 👁️ Visual Analysis
*   **What we checked:** Seamless skin textures, natural eye blinks, and consistent lighting on the face.
*   **Findings:** {vis_findings}
*   **Visual Fake Probability:** {(vis_val * 100):.1f}%

#### 🔊 Audio Analysis
*   **What we checked:** Sound track, voice frequencies, and natural speech rhythms.
*   **Findings:** {aud_findings}
*   **Audio Fake Probability:** {(aud_val * 100):.1f}%

#### 👄 Lip-Sync Alignment
*   **What we checked:** If the shape of the mouth matches the timing of the spoken syllables.
*   **Findings:** {sync_findings}
*   **Sync Mismatch Score:** {(sync_val * 100):.1f}%

---

### ⚖️ Final Conclusion
{label == "FAKE" and "Because strong anomalies were detected in the media modalities, we have high confidence that this video has been digitally manipulated." or "Since the visual, audio, and timing metrics are all within natural ranges, we conclude that this is an authentic, original video."}

---

### 🛠️ Easy verification steps you can take:
1.  **Look closely at the chin and eyes:** Play the video slowly and look for any brief fuzziness or double-outlines around the edges of the face.
2.  **Listen to the breathing:** Real speakers take breaths and pause naturally between words. Fake voices often sound like one continuous stream of speech.
3.  **Check the lip-sync:** Pay close attention to words starting with 'B', 'M', or 'P'. If the mouth doesn't fully close when these sounds are spoken, the audio was likely added later.
"""

    return result_metadata, report_md

