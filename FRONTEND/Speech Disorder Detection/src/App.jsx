import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// API Base URL - points to FastAPI backend (or production backend URL via VITE_API_URL)
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('studio'); // studio | benchmarks | history | dataset
  const [systemHealth, setSystemHealth] = useState({ status: 'connecting', database: 'checking', models_ready: false });
  
  // Studio State
  const [inputMode, setInputMode] = useState('upload'); // 'record' | 'upload'
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  
  // Patient Metadata
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('Male');
  const [clinicalNotes, setClinicalNotes] = useState('');

  // Analysis / Prediction Result State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [screeningResult, setScreeningResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);

  // Benchmarks State
  const [benchmarksData, setBenchmarksData] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainMessage, setTrainMessage] = useState(null);

  // History State
  const [historyRecords, setHistoryRecords] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historySearch, setHistorySearch] = useState('');
  const [selectedDetail, setSelectedDetail] = useState(null);

  // Dataset State
  const [datasetStats, setDatasetStats] = useState({});
  const [generatingData, setGeneratingData] = useState(false);

  // Audio Recording Refs & Monitoring
  const mediaRecorderRef = useRef(null);
  const timerRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const animationFrameRef = useRef(null);
  const monitorGainRef = useRef(null);

  // Live Volume & Monitor State
  const [liveVolume, setLiveVolume] = useState(0);
  const [liveMonitoring, setLiveMonitoring] = useState(false);
  const [recordingWarning, setRecordingWarning] = useState(null);

  // Check health on mount
  useEffect(() => {
    checkHealth();
    fetchBenchmarks();
    fetchHistory();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setSystemHealth(data);
        setDatasetStats(data.data_samples || {});
      } else {
        setSystemHealth({ status: 'offline', database: 'disconnected', models_ready: false });
      }
    } catch {
      setSystemHealth({ status: 'offline', database: 'disconnected', models_ready: false });
    }
  };

  const fetchBenchmarks = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/models/metrics`);
      if (res.ok) {
        const data = await res.json();
        setBenchmarksData(data);
      }
    } catch (e) {
      console.error('Error fetching benchmarks:', e);
    }
  };

  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE}/api/screenings`);
      if (res.ok) {
        const data = await res.json();
        setHistoryRecords(data);
      }
    } catch (e) {
      console.error('Error fetching history:', e);
    } finally {
      setLoadingHistory(false);
    }
  };

// Helper: Resample audio buffer via linear interpolation
function resamplePCM(audioData, origSampleRate, targetSampleRate = 16000) {
  if (origSampleRate === targetSampleRate) return audioData;
  const ratio = origSampleRate / targetSampleRate;
  const newLength = Math.round(audioData.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const origIndex = i * ratio;
    const indexFloor = Math.floor(origIndex);
    const indexCeil = Math.min(audioData.length - 1, indexFloor + 1);
    const frac = origIndex - indexFloor;
    result[i] = audioData[indexFloor] * (1 - frac) + audioData[indexCeil] * frac;
  }
  return result;
}

// Helper: Encode Float32 PCM to 16-bit RIFF WAV with automatic speech level normalization
function encodePCMToWAV(samples, sampleRate = 16000) {
  // Peak normalization so quiet laptop mics are clearly audible on playback
  let maxPeak = 0;
  for (let i = 0; i < samples.length; i++) {
    const absVal = Math.abs(samples[i]);
    if (absVal > maxPeak) maxPeak = absVal;
  }
  // Apply gentle boost if recording is quiet (up to 6x gain), target ~0.85 peak
  const gain = maxPeak > 0.005 ? Math.min(0.85 / maxPeak, 6.0) : 1.0;

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  /* RIFF identifier */
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  /* fmt chunk */
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 = PCM)
  view.setUint16(22, 1, true);  // NumChannels (1 = Mono)
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // ByteRate
  view.setUint16(32, 2, true);  // BlockAlign (1 * 16/8)
  view.setUint16(34, 16, true); // BitsPerSample
  /* data chunk */
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  // Write 16-bit linear PCM with saturation
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i] * gain));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

  // ---------------- MICROPHONE RECORDING LOGIC (MEDIARECORDER + WEBAUDIO DECODER) ----------------
  const streamRef = useRef(null);

  const startRecording = async () => {
    setAnalysisError(null);
    setRecordingWarning(null);
    setAudioBlob(null);
    setAudioUrl(null);
    setSelectedFile(null);
    audioChunksRef.current = [];
    setLiveVolume(0);

    try {
      // 1. Obtain user microphone stream with standard Voice constraints & fallback
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: false,
            autoGainControl: true,
          },
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      streamRef.current = stream;

      // 2. Setup AudioContext for Live Waveform Visualizer & VU Volume Meter
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);

      // Setup Visualizer & RMS Volume Analyser
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.35;
      source.connect(analyser);

      // Setup Optional Live Mic Monitor (routes mic to speakers/headphones if enabled)
      const monitorGain = audioCtx.createGain();
      monitorGain.gain.value = liveMonitoring ? 1 : 0;
      monitorGainRef.current = monitorGain;
      source.connect(monitorGain);
      monitorGain.connect(audioCtx.destination);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      const timeArray = new Float32Array(analyser.fftSize);

      const draw = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        animationFrameRef.current = requestAnimationFrame(draw);

        // Compute Live Volume Level in real time
        analyser.getFloatTimeDomainData(timeArray);
        let peak = 0;
        for (let i = 0; i < timeArray.length; i++) {
          const abs = Math.abs(timeArray[i]);
          if (abs > peak) peak = abs;
        }
        const currentVolPercent = Math.min(100, Math.round(peak * 190));
        setLiveVolume(currentVolPercent);

        // Draw frequency spectrum bars
        analyser.getByteFrequencyData(dataArray);
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * canvas.height * 0.88;
          const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
          gradient.addColorStop(0, '#06b6d4');
          gradient.addColorStop(0.5, '#38bdf8');
          gradient.addColorStop(1, '#a855f7');

          ctx.fillStyle = gradient;
          ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          x += barWidth + 1;
        }
      };
      draw();

      // 3. Setup Standard MediaRecorder (Guaranteed audio capture on Chrome/Edge/Firefox)
      const supportedMimes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
        ''
      ];
      const mime = supportedMimes.find(m => m === '' || (window.MediaRecorder && MediaRecorder.isTypeSupported(m))) || '';
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.start(100); // collect in 100ms chunks

      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setAnalysisError('Microphone access denied or unavailable: ' + err.message);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    clearInterval(timerRef.current);

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = async () => {
        const mimeType = recorder.mimeType || 'audio/webm';
        const rawBlob = new Blob(audioChunksRef.current, { type: mimeType });

        // Stop mic stream tracks to release hardware
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }

        // Try decoding to 16kHz normalized WAV using WebAudio
        try {
          const arrayBuffer = await rawBlob.arrayBuffer();
          const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
          const decodedBuffer = await decodeCtx.decodeAudioData(arrayBuffer);

          const numChannels = decodedBuffer.numberOfChannels;
          const length = decodedBuffer.length;
          const monoPCM = new Float32Array(length);

          if (numChannels === 1) {
            monoPCM.set(decodedBuffer.getChannelData(0));
          } else {
            const ch0 = decodedBuffer.getChannelData(0);
            const ch1 = decodedBuffer.getChannelData(1);
            for (let i = 0; i < length; i++) {
              monoPCM[i] = (ch0[i] + ch1[i]) / 2;
            }
          }

          // Check maximum peak level
          let maxPeak = 0;
          for (let i = 0; i < length; i++) {
            const absVal = Math.abs(monoPCM[i]);
            if (absVal > maxPeak) maxPeak = absVal;
          }

          if (maxPeak < 0.005) {
            setRecordingWarning('Recorded audio appears near silent. Please check that your microphone volume is turned up and unmuted in Windows.');
          } else {
            setRecordingWarning(null);
          }

          // Resample to 16,000 Hz
          const targetSR = 16000;
          const resampledPCM = resamplePCM(monoPCM, decodedBuffer.sampleRate, targetSR);

          // Encode to standard RIFF WAV with automatic speech normalization
          const wavBlob = encodePCMToWAV(resampledPCM, targetSR);
          setAudioBlob(wavBlob);
          const url = URL.createObjectURL(wavBlob);
          setAudioUrl(url);

          decodeCtx.close();
        } catch (err) {
          console.warn('WebAudio decode fallback, using raw MediaRecorder blob:', err);
          setAudioBlob(rawBlob);
          const url = URL.createObjectURL(rawBlob);
          setAudioUrl(url);
        }

        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          try {
            audioContextRef.current.close();
          } catch {}
        }
      };

      recorder.stop();
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    }
  };

  // ---------------- FILE UPLOAD HANDLING ----------------
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setAudioBlob(file);
      setAudioUrl(URL.createObjectURL(file));
      setAnalysisError(null);
    }
  };

  // ---------------- SAMPLE AUDIO TESTER ----------------
  const loadPresetSample = async (category) => {
    setAnalysisError(null);
    setIsAnalyzing(true);
    try {
      const filename = `sample_${category}_01.wav`;
      const res = await fetch(`${API_BASE}/api/audio/${filename}`);
      if (!res.ok) throw new Error('Sample audio not found on server');
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'audio/wav' });
      setSelectedFile(file);
      setAudioBlob(file);
      setAudioUrl(URL.createObjectURL(file));
      setPatientName(`Test Case (${category.toUpperCase()})`);
      setPatientAge(42);
      setClinicalNotes(`Automated acoustic benchmark test for ${category}.`);
    } catch (err) {
      setAnalysisError('Could not load preset sample: ' + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ---------------- RUN SCREENING INFERENCE ----------------
  const handleAnalyze = async () => {
    if (!audioBlob) {
      setAnalysisError('Please record your voice or select a .wav audio file first.');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError(null);
    setScreeningResult(null);

    const formData = new FormData();
    formData.append('audio', audioBlob, selectedFile ? selectedFile.name : 'voice_recording.wav');
    if (patientName) formData.append('patient_name', patientName);
    if (patientAge) formData.append('patient_age', patientAge);
    if (patientGender) formData.append('patient_gender', patientGender);
    if (clinicalNotes) formData.append('notes', clinicalNotes);

    try {
      const response = await fetch(`${API_BASE}/api/screenings/upload-and-screen`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Inference analysis failed.');
      }

      const resultData = await response.json();
      setScreeningResult(resultData);
      fetchHistory(); // Refresh history
      checkHealth();
    } catch (err) {
      setAnalysisError(err.message || 'Network error communicating with the ML server.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ---------------- TRIGGER MODEL RE-TRAINING ----------------
  const handleTrainModels = async () => {
    setIsTraining(true);
    setTrainMessage(null);
    try {
      const res = await fetch(`${API_BASE}/api/models/train`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setTrainMessage(`Model training complete! Best classifier: ${data.data?.best_model}`);
        fetchBenchmarks();
      } else {
        setTrainMessage('Training error: ' + (data.detail || 'Failed'));
      }
    } catch (e) {
      setTrainMessage('Error triggering training: ' + e.message);
    } finally {
      setIsTraining(false);
    }
  };

  // ---------------- GENERATE MOCK DATASET ----------------
  const handleGenerateData = async () => {
    setGeneratingData(true);
    try {
      const res = await fetch(`${API_BASE}/api/dataset/generate-mock`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        alert(data.message);
        checkHealth();
      }
    } catch (e) {
      alert('Error generating dataset: ' + e.message);
    } finally {
      setGeneratingData(false);
    }
  };

  // Helper for disorder badge styling
  const getBadgeClass = (clsName) => {
    const c = (clsName || '').toLowerCase();
    if (c.includes('normal')) return 'badge-normal';
    if (c.includes('dysarthria')) return 'badge-dysarthria';
    if (c.includes('dysphonia')) return 'badge-dysphonia';
    if (c.includes('stutter')) return 'badge-stuttering';
    return 'badge-normal';
  };

  return (
    <div className="app-container">
      {/* ---------------- NAVIGATION HEADER ---------------- */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v20M17 5v14M7 8v8M22 10v4M2 10v4" />
            </svg>
          </div>
          <div>
            <div className="brand-title">
              AcoustiScreen <span className="brand-highlight">AI</span>
            </div>
            <div className="brand-subtitle">
              Speech Disorder Detection System • Dept. of Information Technology (IV B.Tech)
            </div>
          </div>
        </div>

        {/* System & DB Status Badge */}
        <div className="header-meta">
          <div className="status-pill">
            <span className={`status-dot ${systemHealth.status === 'healthy' ? 'online' : 'offline'}`}></span>
            <span>API: {systemHealth.status.toUpperCase()}</span>
            <span className="status-divider">|</span>
            <span>MySQL 3306: {systemHealth.database}</span>
          </div>

          <nav className="header-nav">
            <button
              className={`nav-link ${activeTab === 'studio' ? 'active' : ''}`}
              onClick={() => setActiveTab('studio')}
            >
              Screening Studio
            </button>
            <button
              className={`nav-link ${activeTab === 'benchmarks' ? 'active' : ''}`}
              onClick={() => { setActiveTab('benchmarks'); fetchBenchmarks(); }}
            >
              Model Benchmarks
            </button>
            <button
              className={`nav-link ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => { setActiveTab('history'); fetchHistory(); }}
            >
              Screening History ({historyRecords.length})
            </button>
            <button
              className={`nav-link ${activeTab === 'dataset' ? 'active' : ''}`}
              onClick={() => setActiveTab('dataset')}
            >
              Dataset
            </button>
          </nav>
        </div>
      </header>

      {/* ---------------- MAIN CONTENT AREA ---------------- */}
      <main className="main-content">
        
        {/* ================= TAB 1: SCREENING STUDIO ================= */}
        {activeTab === 'studio' && (
          <div className="studio-layout">
            
            {/* Left Column: Input Panel */}
            <div className="input-column">
              <div className="glass-panel card-section">
                <div className="section-header">
                  <h3>Speech Audio Input</h3>
                  <div className="tab-pill-group">
                    <button
                      className={`pill-btn ${inputMode === 'upload' ? 'active' : ''}`}
                      onClick={() => setInputMode('upload')}
                    >
                      File Upload
                    </button>
                    <button
                      className={`pill-btn ${inputMode === 'record' ? 'active' : ''}`}
                      onClick={() => setInputMode('record')}
                    >
                      Live Microphone
                    </button>
                  </div>
                </div>

                {/* Microphone Recording View */}
                {inputMode === 'record' && (
                  <div className="recording-container">
                    <div className="visualizer-box">
                      <canvas ref={canvasRef} width="420" height="90" className="waveform-canvas" />
                      {isRecording && (
                        <div className="recording-status-overlay">
                          <span className="rec-indicator recording-pulse"></span>
                          <span>Recording: {recordingTime}s</span>
                        </div>
                      )}
                    </div>

                    {/* Real-time Live Mic Volume VU Meter */}
                    {isRecording && (
                      <div className="live-volume-container">
                        <div className="volume-label-row">
                          <span className="vol-title">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
                            Live Mic Volume:
                          </span>
                          <span className={`vol-val ${liveVolume > 10 ? 'active' : 'silent'}`}>
                            {liveVolume > 5 ? `${liveVolume}%` : 'Silent / Speak into Mic'}
                          </span>
                        </div>
                        <div className="volume-track">
                          <div
                            className="volume-fill"
                            style={{
                              width: `${Math.max(liveVolume, 2)}%`,
                              background: liveVolume > 15 ? 'linear-gradient(90deg, #10b981, #06b6d4)' : '#f59e0b'
                            }}
                          ></div>
                        </div>
                      </div>
                    )}

                    {/* Live Mic Monitoring Option */}
                    <div className="monitor-toggle-box">
                      <label className="monitor-label">
                        <input
                          type="checkbox"
                          checked={liveMonitoring}
                          onChange={(e) => {
                            setLiveMonitoring(e.target.checked);
                            if (monitorGainRef.current) {
                              monitorGainRef.current.gain.value = e.target.checked ? 1 : 0;
                            }
                          }}
                        />
                        <span>🎧 Live Mic Monitor (hear your voice in headphones while speaking)</span>
                      </label>
                    </div>

                    <div className="recording-controls">
                      {!isRecording ? (
                        <button className="btn-primary" onClick={startRecording}>
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
                          Start Microphone Recording
                        </button>
                      ) : (
                        <button className="btn-danger" onClick={stopRecording}>
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
                          Stop & Capture Recording
                        </button>
                      )}
                    </div>

                    {recordingWarning && (
                      <div className="warning-callout">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        <span>{recordingWarning}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* File Upload View */}
                {inputMode === 'upload' && (
                  <div className="upload-container">
                    <label className="dropzone-box">
                      <input
                        type="file"
                        accept=".wav,audio/wav"
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                      />
                      <div className="dropzone-content">
                        <div className="upload-icon">
                          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
                        </div>
                        <p className="dropzone-title">
                          {selectedFile ? selectedFile.name : 'Click or Drag .WAV Audio Recording Here'}
                        </p>
                        <span className="dropzone-sub">Supports 16kHz WAV speech samples (mono/stereo)</span>
                      </div>
                    </label>
                  </div>
                )}

                {/* Audio Playback preview */}
                {audioUrl && (
                  <div className="audio-preview-box">
                    <div className="preview-header-row">
                      <span className="preview-label">Captured Audio Sample:</span>
                      <span className="preview-tag">Audible & Normalized (16kHz WAV)</span>
                    </div>
                    <audio controls src={audioUrl} className="custom-audio-player" autoPlay={false} />
                    <span className="preview-hint">Press ▶ Play to listen to your recorded microphone speech sample before analyzing.</span>
                  </div>
                )}

                {/* Quick Presets / Benchmark Samples */}
                <div className="quick-presets">
                  <span className="presets-label">Test with Clinical Benchmark Samples:</span>
                  <div className="preset-buttons">
                    <button className="btn-preset normal" onClick={() => loadPresetSample('normal')}>
                      Healthy Normal
                    </button>
                    <button className="btn-preset dysarthria" onClick={() => loadPresetSample('dysarthria')}>
                      Dysarthria
                    </button>
                    <button className="btn-preset dysphonia" onClick={() => loadPresetSample('dysphonia')}>
                      Dysphonia
                    </button>
                    <button className="btn-preset stuttering" onClick={() => loadPresetSample('stuttering')}>
                      Stuttering
                    </button>
                  </div>
                </div>

                {/* Patient Information Form */}
                <div className="patient-form">
                  <h4 className="form-title">Patient / Subject Information (Saved to MySQL)</h4>
                  <div className="form-grid">
                    <div className="form-group">
                      <label>Patient Full Name</label>
                      <input
                        type="text"
                        placeholder="e.g. A. Sharma"
                        value={patientName}
                        onChange={(e) => setPatientName(e.target.value)}
                        className="form-input"
                      />
                    </div>
                    <div className="form-row-2">
                      <div className="form-group">
                        <label>Age</label>
                        <input
                          type="number"
                          placeholder="e.g. 45"
                          value={patientAge}
                          onChange={(e) => setPatientAge(e.target.value)}
                          className="form-input"
                        />
                      </div>
                      <div className="form-group">
                        <label>Gender</label>
                        <select
                          value={patientGender}
                          onChange={(e) => setPatientGender(e.target.value)}
                          className="form-input"
                        >
                          <option value="Male">Male</option>
                          <option value="Female">Female</option>
                          <option value="Other">Other</option>
                        </select>
                      </div>
                    </div>
                  </div>
                  <div className="form-group" style={{ marginTop: '10px' }}>
                    <label>Clinical Notes</label>
                    <input
                      type="text"
                      placeholder="e.g. Vocal tremor observed, articulatory imprecision"
                      value={clinicalNotes}
                      onChange={(e) => setClinicalNotes(e.target.value)}
                      className="form-input"
                    />
                  </div>
                </div>

                {/* CTA Button */}
                <div className="action-row">
                  <button
                    className="btn-primary btn-large"
                    onClick={handleAnalyze}
                    disabled={isAnalyzing || !audioBlob}
                  >
                    {isAnalyzing ? (
                      <>
                        <span className="spinner"></span>
                        Extracting MFCCs & Running ML Classifiers...
                      </>
                    ) : (
                      <>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
                        Analyze Speech Sample
                      </>
                    )}
                  </button>
                </div>

                {analysisError && (
                  <div className="error-banner">
                    <strong>Error: </strong> {analysisError}
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Screening Results */}
            <div className="result-column">
              {screeningResult ? (
                <div className="glass-panel result-card">
                  <div className="result-header">
                    <div>
                      <span className="result-tag">Preliminary Screening Result</span>
                      <h2 className="result-title">
                        Detected: <span className="highlight-class">{screeningResult.predicted_class.toUpperCase()}</span>
                      </h2>
                    </div>
                    <div className={`badge ${getBadgeClass(screeningResult.predicted_class)}`}>
                      {screeningResult.predicted_class}
                    </div>
                  </div>

                  {/* Confidence Score Meter */}
                  <div className="confidence-meter-box">
                    <div className="confidence-meter-header">
                      <span>Model Confidence Score</span>
                      <span className="confidence-number">{screeningResult.confidence_percentage}</span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{ width: screeningResult.confidence_percentage }}
                      ></div>
                    </div>
                  </div>

                  {/* Probability Breakdown Across All 4 Classes */}
                  <div className="probabilities-card">
                    <h4 className="card-subtitle">Multi-Class Probability Distribution</h4>
                    <div className="prob-list">
                      {Object.entries(screeningResult.probabilities || {}).map(([cls, prob]) => (
                        <div key={cls} className="prob-row">
                          <span className="prob-name">{cls.toUpperCase()}</span>
                          <div className="prob-bar-container">
                            <div
                              className="prob-bar-fill"
                              style={{
                                width: `${Math.max(prob * 100, 2)}%`,
                                background: cls === screeningResult.predicted_class ? 'var(--gradient-brand)' : '#334155'
                              }}
                            ></div>
                          </div>
                          <span className="prob-val">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Key Acoustic Indicators */}
                  <div className="acoustics-card">
                    <h4 className="card-subtitle">Extracted Acoustic & Spectral Markers</h4>
                    <div className="metrics-grid">
                      <div className="metric-box">
                        <span className="metric-label">Mean Pitch (F0)</span>
                        <span className="metric-value">{screeningResult.key_indicators?.mean_pitch_f0_hz || 0} Hz</span>
                        <span className="metric-note">Fundamental Frequency</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Pitch Variability</span>
                        <span className="metric-value">{screeningResult.key_indicators?.pitch_variability_std || 0}</span>
                        <span className="metric-note">F0 Std Dev / Jitter proxy</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">RMS Energy</span>
                        <span className="metric-value">{screeningResult.key_indicators?.energy_rms || 0}</span>
                        <span className="metric-note">Speech Amplitude</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Zero Crossing Rate</span>
                        <span className="metric-value">{screeningResult.key_indicators?.zero_crossing_rate || 0}</span>
                        <span className="metric-note">Turbulence / Voicing</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Spectral Centroid</span>
                        <span className="metric-value">{screeningResult.key_indicators?.spectral_centroid_hz || 0} Hz</span>
                        <span className="metric-note">Center of Spectral Mass</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Classifier Engine</span>
                        <span className="metric-value" style={{ fontSize: '13px', color: '#38bdf8' }}>{screeningResult.model_used}</span>
                        <span className="metric-note">StandardScaler + RBF Kernel</span>
                      </div>
                    </div>
                  </div>

                  {/* Clinical Screening Insight */}
                  <div className="insight-box">
                    <div className="insight-title">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                      Acoustic & Clinical Pattern Analysis
                    </div>
                    <p className="insight-text">{screeningResult.preliminary_screening_note}</p>
                  </div>

                  {/* Medical Disclaimer Alert Banner */}
                  <div className="disclaimer-banner">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    <span>{screeningResult.disclaimer}</span>
                  </div>

                  {/* Database Record ID */}
                  {screeningResult.id && (
                    <div className="db-confirmation">
                      Record successfully stored in MySQL <code>speech_disorder_db.screening_records</code> (Record ID #{screeningResult.id})
                    </div>
                  )}
                </div>
              ) : (
                <div className="glass-panel placeholder-card">
                  <div className="placeholder-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                  </div>
                  <h3>Awaiting Audio Input</h3>
                  <p>Record your voice with the microphone or upload a speech recording (.wav) to perform real-time speech disorder screening.</p>
                  <div className="placeholder-features">
                    <div className="feat-chip">13 MFCC Coefficients</div>
                    <div className="feat-chip">Pitch (F0) Estimation</div>
                    <div className="feat-chip">Energy & Zero Crossing</div>
                    <div className="feat-chip">Spectral Centroid & Rolloff</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================= TAB 2: MODEL BENCHMARKS ================= */}
        {activeTab === 'benchmarks' && (
          <div className="benchmarks-view">
            <div className="glass-panel card-section">
              <div className="section-header">
                <div>
                  <h3>Machine Learning Model Evaluation & Comparison</h3>
                  <p className="section-desc">
                    Stratified 80/20 train/test evaluation across Support Vector Machine (SVM), Random Forest, and Logistic Regression.
                  </p>
                </div>
                <button
                  className="btn-primary"
                  onClick={handleTrainModels}
                  disabled={isTraining}
                >
                  {isTraining ? 'Training In Progress...' : 'Re-Train All Models'}
                </button>
              </div>

              {trainMessage && <div className="info-banner">{trainMessage}</div>}

              {/* Comparison Table */}
              <div className="table-responsive">
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Classifier Algorithm</th>
                      <th>Accuracy</th>
                      <th>Weighted Precision</th>
                      <th>Weighted Recall</th>
                      <th>F1-Score</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarksData?.models_benchmark ? (
                      Object.entries(benchmarksData.models_benchmark).map(([name, metrics]) => {
                        const isBest = name === benchmarksData.best_model || metrics.is_best;
                        return (
                          <tr key={name} className={isBest ? 'highlight-row' : ''}>
                            <td className="model-name-cell">
                              <strong>{name}</strong>
                              {isBest && <span className="best-tag">BEST MODEL</span>}
                            </td>
                            <td>{(metrics.accuracy * 100).toFixed(2)}%</td>
                            <td>{(metrics.precision * 100).toFixed(2)}%</td>
                            <td>{(metrics.recall * 100).toFixed(2)}%</td>
                            <td>
                              <strong style={{ color: '#38bdf8' }}>
                                {(metrics.f1_score * 100).toFixed(2)}%
                              </strong>
                            </td>
                            <td>
                              <span className="badge badge-normal">Trained & Ready</span>
                            </td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan="6" style={{ textAlign: 'center', padding: '30px' }}>
                          No benchmarks available. Click "Re-Train All Models" to run training on the dataset.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Confusion Matrix Viewer */}
              {benchmarksData?.models_benchmark && (
                <div className="cm-section">
                  <h4 className="card-subtitle">Confusion Matrix (Best Model: {benchmarksData.best_model})</h4>
                  <div className="cm-grid">
                    {benchmarksData.classes && (
                      <div className="cm-container">
                        <table className="cm-table">
                          <thead>
                            <tr>
                              <th>Actual \ Pred</th>
                              {benchmarksData.classes.map((cls) => (
                                <th key={cls}>{cls.toUpperCase()}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {benchmarksData.models_benchmark[benchmarksData.best_model]?.confusion_matrix?.map(
                              (row, rowIdx) => (
                                <tr key={rowIdx}>
                                  <th>{benchmarksData.classes[rowIdx].toUpperCase()}</th>
                                  {row.map((val, colIdx) => (
                                    <td
                                      key={colIdx}
                                      className={rowIdx === colIdx ? 'cm-diag' : 'cm-off'}
                                    >
                                      {val}
                                    </td>
                                  ))}
                                </tr>
                              )
                            )}
                          </tbody>
                        </table>
                      </div>
                    )}
                    <div className="cm-legend">
                      <div className="legend-item">
                        <span className="legend-color diag"></span>
                        <span>Correct Classifications (True Positives)</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color off"></span>
                        <span>Misclassifications</span>
                      </div>
                      <p className="cm-note">
                        Evaluated on 16 test samples using 40 standardized acoustic features with cross-validated hyperparameter regularization.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================= TAB 3: SCREENING HISTORY ================= */}
        {activeTab === 'history' && (
          <div className="history-view">
            <div className="glass-panel card-section">
              <div className="section-header">
                <div>
                  <h3>Patient Screening Records</h3>
                  <p className="section-desc">
                    Live historical screening queries and audio playback stored in MySQL (<code>speech_disorder_db</code>).
                  </p>
                </div>
                <input
                  type="text"
                  placeholder="Search by patient name or disorder..."
                  value={historySearch}
                  onChange={(e) => setHistorySearch(e.target.value)}
                  className="search-input"
                />
              </div>

              {loadingHistory ? (
                <div style={{ textAlign: 'center', padding: '40px' }}>Loading MySQL records...</div>
              ) : (
                <div className="table-responsive">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Patient</th>
                        <th>Predicted Condition</th>
                        <th>Confidence</th>
                        <th>Audio Playback</th>
                        <th>Model Used</th>
                      </tr>
                    </thead>
                    <tbody>
                      {historyRecords
                        .filter((r) =>
                          (r.patient_name || '').toLowerCase().includes(historySearch.toLowerCase()) ||
                          (r.predicted_class || '').toLowerCase().includes(historySearch.toLowerCase())
                        )
                        .map((rec) => (
                          <tr key={rec.id}>
                            <td>#{rec.id}</td>
                            <td style={{ color: '#94a3b8', fontSize: '13px' }}>{rec.created_at}</td>
                            <td>
                              <strong>{rec.patient_name}</strong>
                              <div style={{ fontSize: '11px', color: '#64748b' }}>{rec.patient_code}</div>
                            </td>
                            <td>
                              <span className={`badge ${getBadgeClass(rec.predicted_class)}`}>
                                {rec.predicted_class}
                              </span>
                            </td>
                            <td><strong>{rec.confidence_percentage}</strong></td>
                            <td>
                              <audio
                                controls
                                src={`${API_BASE}${rec.audio_url}`}
                                style={{ height: '32px', maxWidth: '200px' }}
                              />
                            </td>
                            <td style={{ fontSize: '12px', color: '#94a3b8' }}>{rec.model_used}</td>
                          </tr>
                        ))}
                      {historyRecords.length === 0 && (
                        <tr>
                          <td colSpan="7" style={{ textAlign: 'center', padding: '40px' }}>
                            No screening records in MySQL database yet. Run an analysis in the Screening Studio!
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================= TAB 4: DATASET ================= */}
        {activeTab === 'dataset' && (
          <div className="dataset-view">
            <div className="glass-panel card-section">
              <div className="section-header">
                <div>
                  <h3>Audio Dataset & Clinical Benchmarks</h3>
                  <p className="section-desc">
                    Raw audio samples categorized into subfolders inside <code>/data</code> for machine learning feature extraction.
                  </p>
                </div>
                <button
                  className="btn-secondary"
                  onClick={handleGenerateData}
                  disabled={generatingData}
                >
                  {generatingData ? 'Generating...' : 'Re-Generate Synthetic Audio Dataset'}
                </button>
              </div>

              {/* Sample Counts Cards */}
              <div className="dataset-cards-grid">
                {Object.entries(datasetStats).map(([clsName, count]) => (
                  <div key={clsName} className="data-stat-card">
                    <span className={`badge ${getBadgeClass(clsName)}`}>{clsName}</span>
                    <div className="stat-number">{count}</div>
                    <span className="stat-label">WAV Audio Files</span>
                    <span className="stat-path">/data/{clsName}/*.wav</span>
                  </div>
                ))}
              </div>

              {/* Academic Guidance */}
              <div className="academic-guidance">
                <h4>Clinical Data Plug-In Guide (For Real Datasets)</h4>
                <p>
                  To plug in real clinical speech recordings into this B.Tech project, simply copy your labeled <code>.wav</code> files into the corresponding directories:
                </p>
                <ul>
                  <li><strong>Saarbruecken Voice Database (SVD):</strong> 2000+ recordings of sustained vowel /a/, /i/, /u/ with vocal cord pathologies (dysphonia, laryngitis). Place in <code>/data/dysphonia/</code>.</li>
                  <li><strong>TORGO Database:</strong> Acoustic and articulatory recordings from speakers with dysarthria. Place in <code>/data/dysarthria/</code>.</li>
                  <li><strong>UASpeech:</strong> Isolated-word dysarthric speech database from the University of Illinois. Place in <code>/data/dysarthria/</code>.</li>
                  <li><strong>UCLASS:</strong> University College London Archive of Stuttered Speech. Place in <code>/data/stuttering/</code>.</li>
                </ul>
                <p>
                  Once you add real files, click <strong>"Re-Train All Models"</strong> in the Model Benchmarks tab to extract features and update the classifier weights.
                </p>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* ---------------- FOOTER ---------------- */}
      <footer className="app-footer">
        <div className="footer-content">
          <span>B.Tech Major Project • Department of Information Technology</span>
          <span className="footer-divider">•</span>
          <span>FastAPI + Scikit-Learn + Librosa + MySQL</span>
          <span className="footer-divider">•</span>
          <span className="footer-alert">Clinical Screening Tool Only — Not a Diagnostic Device</span>
        </div>
      </footer>
    </div>
  );
}
