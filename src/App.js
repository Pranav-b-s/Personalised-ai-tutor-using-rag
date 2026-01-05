import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import "./avatar.css";
import { Routes, Route, useNavigate } from "react-router-dom";
import Roadmap from "./Roadmap";


function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("chat");
  const [studentProfile, setStudentProfile] = useState(null);
  const [interactions, setInteractions] = useState([]);
  const chatBoxRef = useRef(null);
  const navigate = useNavigate();


  // Voice and Avatar states
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [avatarExpression, setAvatarExpression] = useState("neutral");
  const [mouthOpen, setMouthOpen] = useState(0);
  const recognitionRef = useRef(null);
  const synthRef = useRef(window.speechSynthesis);
  const abortControllerRef = useRef(null);
  const listeningTimeoutRef = useRef(null);

  useEffect(() => {
    if (chatBoxRef.current && activeTab === "chat") {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, activeTab]);

  // Initialize Speech Recognition
  useEffect(() => {
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = "en-US";

      recognitionRef.current.onstart = () => {
        setIsListening(true);
        setAvatarExpression("listening");
      };

      recognitionRef.current.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            interim += transcript + " ";
          }
        }

        if (interim.trim()) {
          setQuestion(interim.trim());
        }
      };

      recognitionRef.current.onerror = () => {
        setIsListening(false);
        setAvatarExpression("neutral");
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
        setAvatarExpression("neutral");
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      synthRef.current.cancel();
      if (listeningTimeoutRef.current) {
        clearTimeout(listeningTimeoutRef.current);
      }
    };
  }, []);

  // Animate mouth based on speech
  useEffect(() => {
    if (!isSpeaking) {
      setMouthOpen(0);
      return;
    }

    const interval = setInterval(() => {
      setMouthOpen(() => {
        const variation = Math.sin(Date.now() / 100) * 0.5 + 0.5;
        return variation;
      });
    }, 50);

    return () => clearInterval(interval);
  }, [isSpeaking]);

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start();
      if (listeningTimeoutRef.current) {
        clearTimeout(listeningTimeoutRef.current);
      }
      listeningTimeoutRef.current = setTimeout(() => {
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
      }, 10000);
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      if (listeningTimeoutRef.current) {
        clearTimeout(listeningTimeoutRef.current);
      }
    }
  };

  const speakText = (text) => {
    synthRef.current.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.1;
    utterance.volume = 1;

    utterance.onstart = () => {
      setIsSpeaking(true);
      setAvatarExpression("talking");
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      setAvatarExpression("happy");
      setMouthOpen(0);
      setTimeout(() => setAvatarExpression("neutral"), 1500);
    };

    utterance.onerror = () => {
      setIsSpeaking(false);
      setAvatarExpression("neutral");
      setMouthOpen(0);
    };

    synthRef.current.speak(utterance);
  };

  const stopSpeaking = () => {
    synthRef.current.cancel();
    setIsSpeaking(false);
    setAvatarExpression("neutral");
    setMouthOpen(0);
  };

  const sendQuestion = async (e) => {
    e.preventDefault();

    stopListening();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;

    const userMessage = { sender: "user", text: trimmedQuestion };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);
    setAvatarExpression("thinking");
    stopSpeaking();

    abortControllerRef.current = new AbortController();

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/ask",
        { question: trimmedQuestion },
        {
          timeout: 60000,
          headers: {
            "Content-Type": "application/json",
          },
          signal: abortControllerRef.current.signal,
        }
      );

      const answer = response.data.answer || "No response received.";
      const botMessage = { sender: "bot", text: answer };
      setMessages((prev) => [...prev, botMessage]);

      setLoading(false);
      setAvatarExpression("talking");
      speakText(answer);

      if (activeTab === "profile") {
        fetchProfile();
      }
    } catch (error) {
      if (axios.isCancel(error)) {
        console.log("Request cancelled");
        return;
      }

      let errorMessage = "Unable to connect to tutor.";

      if (error.response) {
        errorMessage =
          error.response.data?.error ||
          `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage =
          "No response from server. Make sure Flask backend is running on port 5000.";
      } else {
        errorMessage = error.message;
      }

      const errorBotMessage = {
        sender: "bot",
        text: `❌ Error: ${errorMessage}`,
      };

      setMessages((prev) => [...prev, errorBotMessage]);
      setAvatarExpression("sad");
      setLoading(false);
    }
  };

  const fetchProfile = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:5000/student-profile");
      setStudentProfile(res.data);
    } catch (err) {
      console.error("Error fetching profile:", err);
      setStudentProfile({ error: "Failed to load profile" });
    }
  };

  const fetchInteractions = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:5000/interactions?limit=20");
      setInteractions(res.data.recent || []);
    } catch (err) {
      console.error("Error fetching interactions:", err);
      setInteractions([]);
    }
  };

  const handleTabChange = (tab) => {
  stopSpeaking();
  stopListening();

  setActiveTab(tab);

  if (tab === "profile") {
    fetchProfile();
    navigate("/"); // ✅ GO BACK TO HOME
  } 
  else if (tab === "history") {
    fetchInteractions();
    navigate("/"); // ✅ GO BACK TO HOME
  } 
  else if (tab === "roadmap") {
    navigate("/roadmap"); // ✅ lowercase + roadmap route
  } 
  else {
    // chat
    navigate("/"); // ✅ GO BACK TO HOME
  }
};
  // 3D Avatar Head Component
  const AvatarHead = ({ expression, mouthOpen, isListening, isSpeaking }) => {
    const headRef = useRef();
    const leftEyeRef = useRef();
    const rightEyeRef = useRef();
    const mouthRef = useRef();
    const blinkTimeRef = useRef(0);

    useFrame((state) => {
      const time = state.clock.getElapsedTime();
      
      blinkTimeRef.current += 0.016;
      if (blinkTimeRef.current > 3) {
        const blinkProgress = (blinkTimeRef.current - 3) / 0.15;
        if (blinkProgress < 1) {
          const scale = 1 - Math.sin(blinkProgress * Math.PI) * 0.9;
          if (leftEyeRef.current) leftEyeRef.current.scale.y = scale;
          if (rightEyeRef.current) rightEyeRef.current.scale.y = scale;
        } else if (blinkProgress >= 1 && blinkTimeRef.current > 3.15) {
          blinkTimeRef.current = 0;
          if (leftEyeRef.current) leftEyeRef.current.scale.y = 1;
          if (rightEyeRef.current) rightEyeRef.current.scale.y = 1;
        }
      }

      if (headRef.current) {
        if (expression === "thinking") {
          headRef.current.rotation.z = -0.1 + Math.sin(time * 0.5) * 0.05;
          headRef.current.rotation.y = Math.sin(time * 0.8) * 0.15;
        } else if (isListening) {
          headRef.current.rotation.z = 0.05;
          headRef.current.rotation.y = Math.sin(time * 2) * 0.05;
        } else {
          headRef.current.rotation.z = Math.sin(time * 0.3) * 0.02;
          headRef.current.rotation.y = Math.sin(time * 0.5) * 0.03;
        }
      }

      if (mouthRef.current && isSpeaking) {
        mouthRef.current.scale.y = 0.3 + mouthOpen * 0.7;
      } else if (mouthRef.current) {
        mouthRef.current.scale.y = 0.3;
      }
    });

    const skinColor = "#ffdbac";
    const eyeColor = expression === "thinking" ? "#4f46e5" : expression === "sad" ? "#ef4444" : "#1e40af";

    return (
      <group ref={headRef}>
        <mesh position={[0, 0, 0]} castShadow>
          <sphereGeometry args={[1, 32, 32]} />
          <meshStandardMaterial color={skinColor} roughness={0.8} metalness={0.1} />
        </mesh>

        <mesh position={[-0.95, 0, 0]} rotation={[0, 0, Math.PI / 6]} castShadow>
          <sphereGeometry args={[0.25, 16, 16]} />
          <meshStandardMaterial color="#ffcba4" roughness={0.9} />
        </mesh>
        <mesh position={[0.95, 0, 0]} rotation={[0, 0, -Math.PI / 6]} castShadow>
          <sphereGeometry args={[0.25, 16, 16]} />
          <meshStandardMaterial color="#ffcba4" roughness={0.9} />
        </mesh>

        <group position={[-0.3, 0.2, 0.8]}>
          <mesh>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial color="white" roughness={0.3} />
          </mesh>
          <mesh ref={leftEyeRef} position={[0, 0, 0.1]}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color={eyeColor} roughness={0.5} />
          </mesh>
          <mesh position={[0.02, 0.02, 0.15]}>
            <sphereGeometry args={[0.03, 8, 8]} />
            <meshStandardMaterial color="white" emissive="white" emissiveIntensity={0.5} />
          </mesh>
        </group>

        <group position={[0.3, 0.2, 0.8]}>
          <mesh>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial color="white" roughness={0.3} />
          </mesh>
          <mesh ref={rightEyeRef} position={[0, 0, 0.1]}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color={eyeColor} roughness={0.5} />
          </mesh>
          <mesh position={[0.02, 0.02, 0.15]}>
            <sphereGeometry args={[0.03, 8, 8]} />
            <meshStandardMaterial color="white" emissive="white" emissiveIntensity={0.5} />
          </mesh>
        </group>

        <mesh position={[-0.3, 0.38, 0.75]} rotation={[0, 0, expression === "thinking" ? -0.2 : expression === "sad" ? 0.2 : 0.1]}>
          <boxGeometry args={[0.25, 0.05, 0.05]} />
          <meshStandardMaterial color="#8b7355" />
        </mesh>
        <mesh position={[0.3, 0.38, 0.75]} rotation={[0, 0, expression === "thinking" ? 0.2 : expression === "sad" ? -0.2 : -0.1]}>
          <boxGeometry args={[0.25, 0.05, 0.05]} />
          <meshStandardMaterial color="#8b7355" />
        </mesh>

        <mesh position={[0, 0, 0.95]} rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.1, 0.3, 8]} />
          <meshStandardMaterial color="#ffcba4" roughness={0.8} />
        </mesh>

        <mesh ref={mouthRef} position={[0, -0.25, 0.85]}>
          <sphereGeometry args={[0.15, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial 
            color={expression === "happy" ? "#ff6b9d" : "#d97676"} 
            side={THREE.DoubleSide}
            roughness={0.6}
          />
        </mesh>

        {(expression === "happy" || isSpeaking) && (
          <>
            <mesh position={[-0.6, -0.1, 0.6]}>
              <sphereGeometry args={[0.15, 16, 16]} />
              <meshStandardMaterial color="#ff9999" transparent opacity={0.4} />
            </mesh>
            <mesh position={[0.6, -0.1, 0.6]}>
              <sphereGeometry args={[0.15, 16, 16]} />
              <meshStandardMaterial color="#ff9999" transparent opacity={0.4} />
            </mesh>
          </>
        )}

        <mesh position={[0, -1.2, 0]} castShadow>
          <cylinderGeometry args={[0.35, 0.4, 0.5, 16]} />
          <meshStandardMaterial color={skinColor} roughness={0.8} />
        </mesh>

        <mesh position={[0, -1.8, 0]} castShadow>
          <cylinderGeometry args={[0.75, 0.9, 1.2, 16]} />
          <meshStandardMaterial color="#3b82f6" roughness={0.7} />
        </mesh>

        <mesh position={[0, -1.4, 0.3]} rotation={[0.3, 0, 0]} castShadow>
          <boxGeometry args={[0.6, 0.15, 0.1]} />
          <meshStandardMaterial color="#2563eb" roughness={0.6} />
        </mesh>

        {[0, -0.3, -0.6].map((yOffset, i) => (
          <mesh key={i} position={[0, -1.5 + yOffset, 0.72]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshStandardMaterial color="white" roughness={0.4} metalness={0.3} />
          </mesh>
        ))}

        <mesh position={[0, 0.6, -0.1]} castShadow>
          <sphereGeometry args={[0.9, 32, 32, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial color="#3d2817" roughness={0.95} />
        </mesh>
      </group>
    );
  };

  const ListeningParticles = () => {
    const particlesRef = useRef();
    
    useFrame((state) => {
      if (particlesRef.current) {
        particlesRef.current.rotation.y = state.clock.getElapsedTime() * 0.5;
      }
    });

    return (
      <group ref={particlesRef}>
        {[...Array(20)].map((_, i) => {
          const angle = (i / 20) * Math.PI * 2;
          const radius = 2.5;
          return (
            <mesh
              key={i}
              position={[
                Math.cos(angle) * radius,
                Math.sin(i * 0.5) * 0.5,
                Math.sin(angle) * radius,
              ]}
            >
              <sphereGeometry args={[0.05, 8, 8]} />
              <meshStandardMaterial 
                color="#10b981" 
                emissive="#10b981" 
                emissiveIntensity={1}
              />
            </mesh>
          );
        })}
      </group>
    );
  };

  const SpeakingWaves = () => {
    const wavesRef = useRef([]);
    
    useFrame((state) => {
      wavesRef.current.forEach((wave, i) => {
        if (wave) {
          wave.rotation.x = state.clock.getElapsedTime() * (0.5 + i * 0.2);
          wave.rotation.y = state.clock.getElapsedTime() * (0.3 + i * 0.1);
        }
      });
    });

    return (
      <group>
        {[1.2, 1.6, 2.0].map((radius, i) => (
          <mesh
            key={i}
            ref={(el) => (wavesRef.current[i] = el)}
            position={[0, 0, 0]}
          >
            <torusGeometry args={[radius, 0.03, 16, 100]} />
            <meshStandardMaterial
              color="#10b981"
              transparent
              opacity={0.4 - i * 0.1}
              emissive="#10b981"
              emissiveIntensity={0.5}
            />
          </mesh>
        ))}
      </group>
    );
  };

  const RealisticAvatar = () => {
    return (
      <div className="avatar-container">
        <Canvas
          camera={{ position: [0, 0, 4], fov: 50 }}
          className="avatar-canvas"
          shadows
        >
          <PerspectiveCamera makeDefault position={[0, 0, 4]} />
          
          <ambientLight intensity={0.6} />
          <directionalLight 
            position={[5, 5, 5]} 
            intensity={0.8} 
            castShadow 
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
          />
          <pointLight position={[-5, 3, -5]} intensity={0.4} color="#ffd6a5" />
          <spotLight 
            position={[0, 5, 0]} 
            intensity={0.5} 
            angle={0.6} 
            penumbra={1}
            castShadow
          />

          <AvatarHead
            expression={avatarExpression}
            mouthOpen={mouthOpen}
            isListening={isListening}
            isSpeaking={isSpeaking}
          />

          {isListening && <ListeningParticles />}
          {isSpeaking && <SpeakingWaves />}

          <OrbitControls
            enableZoom={false}
            enablePan={false}
            minPolarAngle={Math.PI / 2.5}
            maxPolarAngle={Math.PI / 1.5}
            minAzimuthAngle={-Math.PI / 4}
            maxAzimuthAngle={Math.PI / 4}
          />
        </Canvas>

        {avatarExpression === "thinking" && (
          <div className="thinking-indicator">
            <div className="dot" style={{animationDelay: "0s"}} />
            <div className="dot" style={{animationDelay: "0.2s"}} />
            <div className="dot" style={{animationDelay: "0.4s"}} />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-wrapper">
      <nav className="navbar">
        <h1 className="navbar-title">🎓 AI Virtual Tutor - Bob</h1>
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => handleTabChange("chat")}
          >
            💬 Chat
          </button>
          <button
            className={`tab-button ${activeTab === "profile" ? "active" : ""}`}
            onClick={() => handleTabChange("profile")}
          >
            👤 Profile
          </button>
          <button
            className={`tab-button ${activeTab === "history" ? "active" : ""}`}
            onClick={() => handleTabChange("history")}
          >
            📚 History
          </button>
          <button
            className={`tab-button ${activeTab === "roadmap" ? "active" : ""}`}
            onClick={() => handleTabChange("roadmap")}
          >
            🗺️ Roadmap
          </button>
        </div>
      </nav>

      <div className="main-content">
        <Routes>
          {/* HOME / TABS */}
          <Route
            path="/"
            element={
              <>
                {/* CHAT TAB */}
                {activeTab === "chat" && (
                  <div className="chat-layout">
                    <aside className="avatar-sidebar">
                      <RealisticAvatar />

                      <div className="voice-controls">
                        <button
                          onClick={isListening ? stopListening : startListening}
                          className={`control-button ${isListening ? "listening" : ""}`}
                          disabled={loading || isSpeaking}
                        >
                          {isListening ? "🎤 Listening..." : "🎤 Press to Speak"}
                        </button>

                        {isSpeaking && (
                          <button
                            onClick={stopSpeaking}
                            className="control-button stop-speaking"
                          >
                            ⏹️ Stop Speaking
                          </button>
                        )}
                      </div>

                      <div className="status-display">
                        {loading && "🤔 Bob is thinking..."}
                        {isSpeaking && "🗣️ Bob is speaking..."}
                        {isListening && "👂 Listening to you..."}
                        {!loading && !isSpeaking && !isListening && "💬 Ready to help!"}
                      </div>
                    </aside>

                    <div className="chat-area">
                      <div className="chat-messages" ref={chatBoxRef}>
                        {messages.length === 0 && (
                          <div className="empty-chat">
                            <p>👋 Hi! I'm Bob, your AI tutor.</p>
                            <p>Click the microphone and start speaking!</p>
                          </div>
                        )}

                        {messages.map((msg, i) => (
                          <div key={i} className={`message-wrapper ${msg.sender}`}>
                            <div className={`message ${msg.sender}`}>
                              {msg.sender === "bot" && (
                                <button
                                  onClick={() => speakText(msg.text)}
                                  className="speak-button"
                                >
                                  🔊
                                </button>
                              )}
                              {msg.text}
                            </div>
                          </div>
                        ))}

                        {loading && (
                          <div className="loading-indicator">
                            <span className="loading-dot">●</span>
                            <span className="loading-dot">●</span>
                            <span className="loading-dot">●</span>
                          </div>
                        )}
                      </div>

                      <form onSubmit={sendQuestion} className="chat-input-form">
                        <input
                          type="text"
                          placeholder="Or type your question..."
                          value={question}
                          onChange={(e) => setQuestion(e.target.value)}
                          className="chat-input"
                          disabled={loading}
                        />
                        <button type="submit" className="send-button" disabled={loading}>
                          {loading ? "⏳" : "Send"}
                        </button>
                      </form>
                    </div>
                  </div>
                )}

                {/* PROFILE TAB */}
                {activeTab === "profile" && (
                  <div className="tab-content">
                    {!studentProfile ? (
                      <div className="loading-state">Loading profile...</div>
                    ) : studentProfile.error ? (
                      <div className="error-state">❌ {studentProfile.error}</div>
                    ) : (
                      <>
                        <h3 className="section-title">📊 Your Learning Profile</h3>

                        <div className="stat-card">
                          <div className="stat-label">Total Interactions</div>
                          <div className="stat-value">
                            {studentProfile.total_interactions || 0}
                          </div>
                        </div>

                        <button onClick={fetchProfile} className="refresh-button">
                          🔄 Refresh Profile
                        </button>
                      </>
                    )}
                  </div>
                )}

                {/* HISTORY TAB */}
                {activeTab === "history" && (
                  <div className="tab-content">
                    <h3 className="section-title">📚 Your Learning History</h3>

                    {interactions.length === 0 ? (
                      <div className="empty-state">
                        No interactions yet. Start chatting!
                      </div>
                    ) : (
                      <div className="history-list">
                        {interactions.map((item, i) => (
                          <div key={i} className="history-item">
                            <strong>Q:</strong> {item.question}
                            <br />
                            <strong>A:</strong> {item.answer}
                          </div>
                        ))}
                      </div>
                    )}

                    <button onClick={fetchInteractions} className="refresh-button">
                      🔄 Refresh History
                    </button>
                  </div>
                )}
              </>
            }
          />

          {/* ROADMAP PAGE */}
          <Route path="/roadmap" element={<Roadmap />} />
        </Routes>
      </div>


      <footer className="footer">Powered by AI & React Three Fiber</footer>
    </div>
  );
}

export default App;