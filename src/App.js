import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("chat");
  const [studentProfile, setStudentProfile] = useState(null);
  const [interactions, setInteractions] = useState([]);
  const chatBoxRef = useRef(null);
  
  // Roadmap states
  const [roadmapGoal, setRoadmapGoal] = useState("");
  const [studyTime, setStudyTime] = useState("1");
  const [learningStyle, setLearningStyle] = useState("visual");
  const [roadmapLoading, setRoadmapLoading] = useState(false);
  const [roadmapResult, setRoadmapResult] = useState(null);

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
      setMouthOpen((prev) => {
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

  const generateRoadmap = async () => {
    if (!roadmapGoal.trim()) {
      alert("Please enter a learning goal");
      return;
    }

    setRoadmapLoading(true);
    setRoadmapResult(null);

    try {
      const res = await axios.post(
        "http://127.0.0.1:5000/learning-roadmap",
        {
          goal: roadmapGoal,
          time_per_day: studyTime,
          learning_style: learningStyle,
        },
        {
          headers: { "Content-Type": "application/json" },
        }
      );

      setRoadmapResult(res.data);
    } catch (err) {
      console.error("Roadmap error:", err);
      setRoadmapResult({ error: "Failed to generate roadmap" });
    } finally {
      setRoadmapLoading(false);
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    stopSpeaking();
    stopListening();
    if (tab === "profile") {
      fetchProfile();
    } else if (tab === "history") {
      fetchInteractions();
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
      
      // Blink animation
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

      // Head movements
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

      // Mouth animation
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
        {/* Head */}
        <mesh position={[0, 0, 0]} castShadow>
          <sphereGeometry args={[1, 32, 32]} />
          <meshStandardMaterial color={skinColor} roughness={0.8} metalness={0.1} />
        </mesh>

        {/* Ears */}
        <mesh position={[-0.95, 0, 0]} rotation={[0, 0, Math.PI / 6]} castShadow>
          <sphereGeometry args={[0.25, 16, 16]} />
          <meshStandardMaterial color="#ffcba4" roughness={0.9} />
        </mesh>
        <mesh position={[0.95, 0, 0]} rotation={[0, 0, -Math.PI / 6]} castShadow>
          <sphereGeometry args={[0.25, 16, 16]} />
          <meshStandardMaterial color="#ffcba4" roughness={0.9} />
        </mesh>

        {/* Eyes - Left */}
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

        {/* Eyes - Right */}
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

        {/* Eyebrows */}
        <mesh position={[-0.3, 0.38, 0.75]} rotation={[0, 0, expression === "thinking" ? -0.2 : expression === "sad" ? 0.2 : 0.1]}>
          <boxGeometry args={[0.25, 0.05, 0.05]} />
          <meshStandardMaterial color="#8b7355" />
        </mesh>
        <mesh position={[0.3, 0.38, 0.75]} rotation={[0, 0, expression === "thinking" ? 0.2 : expression === "sad" ? -0.2 : -0.1]}>
          <boxGeometry args={[0.25, 0.05, 0.05]} />
          <meshStandardMaterial color="#8b7355" />
        </mesh>

        {/* Nose */}
        <mesh position={[0, 0, 0.95]} rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.1, 0.3, 8]} />
          <meshStandardMaterial color="#ffcba4" roughness={0.8} />
        </mesh>

        {/* Mouth */}
        <mesh ref={mouthRef} position={[0, -0.25, 0.85]}>
          <sphereGeometry args={[0.15, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial 
            color={expression === "happy" ? "#ff6b9d" : "#d97676"} 
            side={THREE.DoubleSide}
            roughness={0.6}
          />
        </mesh>

        {/* Cheeks (blush) */}
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

        {/* Neck */}
        <mesh position={[0, -1.2, 0]} castShadow>
          <cylinderGeometry args={[0.35, 0.4, 0.5, 16]} />
          <meshStandardMaterial color={skinColor} roughness={0.8} />
        </mesh>

        {/* Shirt */}
        <mesh position={[0, -1.8, 0]} castShadow>
          <cylinderGeometry args={[0.75, 0.9, 1.2, 16]} />
          <meshStandardMaterial color="#3b82f6" roughness={0.7} />
        </mesh>

        {/* Shirt Collar */}
        <mesh position={[0, -1.4, 0.3]} rotation={[0.3, 0, 0]} castShadow>
          <boxGeometry args={[0.6, 0.15, 0.1]} />
          <meshStandardMaterial color="#2563eb" roughness={0.6} />
        </mesh>

        {/* Shirt Buttons */}
        {[0, -0.3, -0.6].map((yOffset, i) => (
          <mesh key={i} position={[0, -1.5 + yOffset, 0.72]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshStandardMaterial color="white" roughness={0.4} metalness={0.3} />
          </mesh>
        ))}

        {/* Hair */}
        <mesh position={[0, 0.6, -0.1]} castShadow>
          <sphereGeometry args={[0.9, 32, 32, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial color="#3d2817" roughness={0.95} />
        </mesh>
      </group>
    );
  };

  // Listening particles
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

  // Speaking waves
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

  // 3D Avatar Scene
  const RealisticAvatar = () => {
    return (
      <div className="flex flex-col justify-center items-center w-full relative">
        <Canvas
          camera={{ position: [0, 0, 4], fov: 50 }}
          className="w-full h-[350px] rounded-2xl"
          shadows
        >
          <PerspectiveCamera makeDefault position={[0, 0, 4]} />
          
          {/* Lighting */}
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

          {/* Avatar */}
          <AvatarHead
            expression={avatarExpression}
            mouthOpen={mouthOpen}
            isListening={isListening}
            isSpeaking={isSpeaking}
          />

          {/* Effects */}
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

        {/* Status indicators */}
        {avatarExpression === "thinking" && (
          <div className="absolute bottom-5 flex gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-indigo-600 animate-bounce" style={{animationDelay: "0s"}} />
            <div className="w-2.5 h-2.5 rounded-full bg-indigo-600 animate-bounce" style={{animationDelay: "0.2s"}} />
            <div className="w-2.5 h-2.5 rounded-full bg-indigo-600 animate-bounce" style={{animationDelay: "0.4s"}} />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="font-sans max-w-7xl mx-auto my-5 p-8 rounded-2xl shadow-xl bg-white">
      <h2 className="text-center mb-5 text-gray-800 text-3xl font-bold">🎓 AI Virtual Tutor - Bob</h2>

      {/* Tab Navigation */}
      <div className="flex gap-1 border-b-2 border-gray-200 mb-5">
        <button
          className={`flex-1 py-3 px-5 bg-transparent border-none border-b-3 cursor-pointer text-base font-medium transition-all ${
            activeTab === "chat" ? "text-blue-600 border-b-blue-600 font-semibold" : "text-gray-500 border-b-transparent"
          }`}
          onClick={() => handleTabChange("chat")}
        >
          💬 Chat
        </button>
        <button
          className={`flex-1 py-3 px-5 bg-transparent border-none border-b-3 cursor-pointer text-base font-medium transition-all ${
            activeTab === "profile" ? "text-blue-600 border-b-blue-600 font-semibold" : "text-gray-500 border-b-transparent"
          }`}
          onClick={() => handleTabChange("profile")}
        >
          👤 Your Profile
        </button>
        <button
          className={`flex-1 py-3 px-5 bg-transparent border-none border-b-3 cursor-pointer text-base font-medium transition-all ${
            activeTab === "history" ? "text-blue-600 border-b-blue-600 font-semibold" : "text-gray-500 border-b-transparent"
          }`}
          onClick={() => handleTabChange("history")}
        >
          📚 Learning History
        </button>
        <button
          className={`flex-1 py-3 px-5 bg-transparent border-none border-b-3 cursor-pointer text-base font-medium transition-all ${
            activeTab === "roadmap" ? "text-blue-600 border-b-blue-600 font-semibold" : "text-gray-500 border-b-transparent"
          }`}
          onClick={() => handleTabChange("roadmap")}
        >
          🗺️ Learning Roadmap
        </button>
      </div>

      {/* Tab Content */}
      <div className="min-h-[600px]">
        {/* CHAT TAB with Realistic Avatar */}
        {activeTab === "chat" && (
          <div className="grid grid-cols-[380px_1fr] gap-8 items-start">
            {/* Avatar Section */}
            <div className="flex flex-col items-center gap-5 p-5 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-3xl shadow-lg">
              <RealisticAvatar />

              {/* Voice Controls */}
              <div className="flex flex-col gap-2.5 w-full">
                <button
                  onClick={isListening ? stopListening : startListening}
                  className={`py-3 px-5 border-none rounded-xl text-base font-semibold text-white cursor-pointer transition-all shadow-md ${
                    isListening ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
                  }`}
                  disabled={loading || isSpeaking}
                >
                  {isListening ? "🎤 Listening..." : "🎤 Press to Speak"}
                </button>

                {isSpeaking && (
                  <button
                    onClick={stopSpeaking}
                    className="py-3 px-5 border-none rounded-xl text-base font-semibold text-white cursor-pointer transition-all shadow-md bg-amber-500 hover:bg-amber-600"
                  >
                    ⏹️ Stop Speaking
                  </button>
                )}
              </div>

              {/* Status */}
              <div className="text-white text-sm font-medium text-center py-2.5 bg-white/20 rounded-xl min-h-[40px] flex items-center justify-center w-full">
                {loading && "🤔 Bob is thinking..."}
                {isSpeaking && "🗣️ Bob is speaking..."}
                {isListening && "👂 Listening to you..."}
                {!loading && !isSpeaking && !isListening && "💬 Ready to help!"}
              </div>
            </div>

            {/* Chat Section */}
            <div className="flex flex-col h-[600px]">
              <div 
                className="flex flex-col gap-3 flex-1 overflow-y-auto border-2 border-gray-200 rounded-xl p-4 mb-5 bg-gray-50"
                ref={chatBoxRef}
              >
                {messages.length === 0 && (
                  <div className="text-center text-gray-400 mt-[200px] text-base">
                    <p>👋 Hi! I'm Bob, your AI tutor.</p>
                    <p>Click the microphone and start speaking!</p>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex w-full ${
                      msg.sender === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`py-3 px-4 rounded-2xl max-w-[75%] break-words text-base leading-relaxed shadow-sm relative ${
                        msg.sender === "user" 
                          ? "bg-blue-600 text-white" 
                          : "bg-gray-100 text-black"
                      }`}
                    >
                      {msg.sender === "bot" && (
                        <button
                          onClick={() => speakText(msg.text)}
                          className="absolute top-1 right-1 bg-transparent border-none cursor-pointer text-base opacity-70 hover:opacity-100 transition-opacity"
                          title="Hear this response"
                        >
                          🔊
                        </button>
                      )}
                      {msg.text}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex items-center gap-2.5 p-2.5">
                    <div className="flex gap-1">
                      <span className="text-xl text-blue-600 animate-bounce">●</span>
                      <span className="text-xl text-blue-600 animate-bounce" style={{animationDelay: "0.2s"}}>●</span>
                      <span className="text-xl text-blue-600 animate-bounce" style={{animationDelay: "0.4s"}}>●</span>
                    </div>
                  </div>
                )}
              </div>

              <form onSubmit={sendQuestion} className="flex gap-3">
                <input
                  type="text"
                  placeholder="Or type your question..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  className="flex-1 py-3.5 px-4 rounded-xl border-2 border-gray-200 text-base outline-none focus:border-blue-500"
                  disabled={loading}
                />
                <button
                  type="submit"
                  className="py-3.5 px-8 bg-blue-600 text-white border-none rounded-xl text-base font-semibold cursor-pointer hover:bg-blue-700 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                  disabled={loading}
                >
                  {loading ? "⏳" : "Send"}
                </button>
              </form>
            </div>
          </div>
        )}

        {/* PROFILE TAB */}
        {activeTab === "profile" && (
          <div className="p-5 overflow-y-auto max-h-[600px]">
            {!studentProfile ? (
              <div className="text-center p-10 text-gray-600">
                <p>Loading profile...</p>
              </div>
            ) : studentProfile.error ? (
              <div className="bg-red-50 text-red-700 p-5 rounded-xl text-center">
                <p>❌ {studentProfile.error}</p>
              </div>
            ) : (
              <>
                <h3 className="text-gray-800 mb-5 text-2xl font-bold">📊 Your Learning Profile</h3>

                <div className="bg-gray-50 p-4 rounded-xl mb-4 border border-gray-200">
                  <div className="text-sm text-gray-600 mb-1">Total Interactions</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {studentProfile.total_interactions || 0}
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-xl mb-4 border border-gray-200">
                  <div className="text-sm text-gray-600 mb-1">Learning Since</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {studentProfile.first_interaction
                      ? new Date(studentProfile.first_interaction).toLocaleDateString()
                      : "N/A"}
                  </div>
                </div>

                <h4 className="text-gray-700 mt-6 mb-4 text-xl font-semibold">📚 Topics You've Explored</h4>
                <div className="flex flex-wrap gap-2.5 mb-5">
                  {studentProfile.topics_discussed &&
                  Object.keys(studentProfile.topics_discussed).length > 0 ? (
                    Object.entries(studentProfile.topics_discussed)
                      .sort((a, b) => b[1] - a[1])
                      .map(([type, count]) => (
                        <div key={type} className="flex items-center gap-2.5 mb-2.5 w-full">
                          <span className="flex-[0_0_180px] text-sm text-gray-700">
                            {type
                              .replace(/_/g, " ")
                              .replace(/\b\w/g, (l) => l.toUpperCase())}
                          </span>
                          <span className="flex-1 h-5 bg-gray-200 rounded-xl overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-300"
                              style={{
                                width: `${(count / studentProfile.total_interactions) * 100}%`,
                              }}
                            />
                          </span>
                          <span className="flex-[0_0_40px] text-right font-bold text-blue-600">{count}</span>
                        </div>
                      ))
                  ) : (
                    <p className="text-center text-gray-400 py-10 px-5 text-base">No data yet.</p>
                  )}
                </div>

                <button onClick={fetchProfile} className="mt-5 py-2.5 px-5 bg-blue-600 text-white border-none rounded-lg cursor-pointer text-sm font-semibold hover:bg-blue-700 transition-colors">
                  🔄 Refresh Profile
                </button>
              </>
            )}
          </div>
        )}

        {/* HISTORY TAB */}
        {activeTab === "history" && (
          <div className="p-5 overflow-y-auto max-h-[600px]">
            <h3 className="text-gray-800 mb-5 text-2xl font-bold">📚 Your Learning History</h3>
            <p className="text-gray-600 text-sm mb-5">
              Recent conversations (showing last 20)
            </p>

            {interactions.length === 0 ? (
              <div className="text-center text-gray-400 py-10 px-5 text-base">
                <p>No interactions yet. Start chatting with Bob!</p>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {interactions
                  .slice()
                  .reverse()
                  .map((interaction, idx) => (
                    <div key={idx} className="bg-gray-50 p-4 rounded-xl border border-gray-200">
                      <div className="text-xs text-gray-400 mb-2.5">
                        🕒 {new Date(interaction.timestamp).toLocaleString()}
                      </div>
                      <div className="mb-2.5 text-sm text-gray-800">
                        <strong>Q:</strong> {interaction.question}
                      </div>
                      <div className="text-sm text-gray-600 mb-2.5 pl-2.5 border-l-3 border-blue-600">
                        <strong>A:</strong> {interaction.answer}
                      </div>
                      {interaction.topics && interaction.topics.length > 0 && (
                        <div className="flex gap-1 flex-wrap mt-2.5">
                          {interaction.topics.map((topic, i) => (
                            <span key={i} className="bg-blue-50 text-blue-600 py-1 px-2.5 rounded-xl text-xs">
                              {topic}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            )}

            <button onClick={fetchInteractions} className="mt-5 py-2.5 px-5 bg-blue-600 text-white border-none rounded-lg cursor-pointer text-sm font-semibold hover:bg-blue-700 transition-colors">
              🔄 Refresh History
            </button>
          </div>
        )}

        {/* ROADMAP TAB */}
        {activeTab === "roadmap" && (
          <div className="p-5 overflow-y-auto max-h-[600px]">
            <h3 className="text-gray-800 mb-5 text-2xl font-bold">🗺️ Personalized Learning Roadmap</h3>

            <input
              className="flex-1 py-3.5 px-4 rounded-xl border-2 border-gray-200 text-base outline-none focus:border-blue-500 w-full mb-4"
              placeholder="What do you want to learn?"
              value={roadmapGoal}
              onChange={(e) => setRoadmapGoal(e.target.value)}
            />

            <select
              className="flex-1 py-3.5 px-4 rounded-xl border-2 border-gray-200 text-base outline-none focus:border-blue-500 w-full mb-4"
              value={studyTime}
              onChange={(e) => setStudyTime(e.target.value)}
            >
              <option value="0.5">30 minutes / day</option>
              <option value="1">1 hour / day</option>
              <option value="2">2 hours / day</option>
              <option value="3">3+ hours / day</option>
            </select>

            <select
              className="flex-1 py-3.5 px-4 rounded-xl border-2 border-gray-200 text-base outline-none focus:border-blue-500 w-full mb-4"
              value={learningStyle}
              onChange={(e) => setLearningStyle(e.target.value)}
            >
              <option value="visual">Visual</option>
              <option value="reading">Reading</option>
              <option value="kinesthetic">Hands-on</option>
              <option value="auditory">Auditory</option>
            </select>

            <button 
              onClick={generateRoadmap} 
              className="py-3.5 px-8 bg-blue-600 text-white border-none rounded-xl text-base font-semibold cursor-pointer hover:bg-blue-700 transition-colors w-full mb-4"
            >
              🚀 Generate Roadmap
            </button>

            {roadmapLoading && <p className="text-center text-gray-600">⏳ Generating roadmap...</p>}

            {roadmapResult && roadmapResult.error && (
              <p className="text-red-600 text-center">{roadmapResult.error}</p>
            )}

            {roadmapResult && !roadmapResult.error && (
              <div className="bg-gray-50 p-4 rounded-xl mb-4 border border-gray-200">
                <h4 className="text-lg font-semibold mb-3">📅 Weekly Plan</h4>
                <ul className="list-disc pl-5 mb-4">
                  {roadmapResult.weekly_plan.map((week, i) => (
                    <li key={i} className="mb-2">{week}</li>
                  ))}
                </ul>

                <h4 className="text-lg font-semibold mb-3">🎯 Study Pattern</h4>
                <p className="mb-4">{roadmapResult.study_pattern}</p>

                <h4 className="text-lg font-semibold mb-3">📺 Recommended Resources</h4>
                <ul className="list-disc pl-5">
                  {roadmapResult.resources.map((r, i) => (
                    <li key={i} className="mb-2">
                      <a href={r.link} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                        {r.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      <p className="text-center text-xs text-gray-400 mt-5">Powered by AI & React Three Fiber</p>
    </div>
  );
}

export default App;