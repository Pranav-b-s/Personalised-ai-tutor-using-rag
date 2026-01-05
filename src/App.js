import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

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
      // Auto-stop after 10 seconds of silence
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

  // Realistic 3D-style Avatar Component
  const RealisticAvatar = () => {
    const eyeBlinkAnimation = `
      @keyframes blink {
        0%, 49%, 51%, 100% { scaleY: 1; }
        50% { scaleY: 0.1; }
      }
    `;

    const getEyeColor = () => {
      switch (avatarExpression) {
        case "thinking":
          return "#4f46e5";
        case "sad":
          return "#ef4444";
        default:
          return "#1e40af";
      }
    };

    const getMouthPath = () => {
      const open = mouthOpen;
      if (isSpeaking) {
        return `M 95 95 Q 100 ${95 + open * 15} 105 95`;
      }

      switch (avatarExpression) {
        case "happy":
          return "M 90 90 Q 100 105 110 90";
        case "sad":
          return "M 90 110 Q 100 95 110 110";
        case "thinking":
          return "M 95 100 L 105 100";
        default:
          return "M 95 100 L 105 100";
      }
    };

    const getHeadTilt = () => {
      if (avatarExpression === "thinking") return "rotate(-5deg)";
      if (isListening) return "rotate(3deg)";
      return "rotate(0deg)";
    };

    return (
      <div style={styles.avatarContainer}>
        <style>{eyeBlinkAnimation}</style>
        <svg
          width="300"
          height="350"
          viewBox="0 0 200 280"
          style={{
            filter: "drop-shadow(0 10px 30px rgba(0,0,0,0.3))",
            transform: getHeadTilt(),
            transition: "transform 0.3s ease",
          }}
        >
          {/* Head - Gradient */}
          <defs>
            <linearGradient
              id="headGradient"
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#fdbcb4" />
              <stop offset="100%" stopColor="#f5a593" />
            </linearGradient>
            <radialGradient id="highlight" cx="30%" cy="30%">
              <stop offset="0%" stopColor="white" stopOpacity="0.6" />
              <stop offset="100%" stopColor="white" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Head */}
          <ellipse cx="100" cy="80" rx="55" ry="65" fill="url(#headGradient)" />

          {/* Shine */}
          <ellipse cx="75" cy="50" rx="25" ry="35" fill="url(#highlight)" />

          {/* Ears */}
          <ellipse
            cx="50"
            cy="70"
            rx="15"
            ry="25"
            fill="#f5a593"
            opacity="0.8"
          />
          <ellipse
            cx="150"
            cy="70"
            rx="15"
            ry="25"
            fill="#f5a593"
            opacity="0.8"
          />

          {/* Eyes */}
          <g>
            {/* Left Eye */}
            <ellipse
              cx="75"
              cy="70"
              rx="12"
              ry="16"
              fill="white"
              style={{
                filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.1))",
              }}
            />
            <circle
              cx="75"
              cy="72"
              r="8"
              fill={getEyeColor()}
              style={{ animation: "blink 4s infinite" }}
            />
            <circle cx="76" cy="70" r="3" fill="white" />

            {/* Right Eye */}
            <ellipse
              cx="125"
              cy="70"
              rx="12"
              ry="16"
              fill="white"
              style={{
                filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.1))",
              }}
            />
            <circle
              cx="125"
              cy="72"
              r="8"
              fill={getEyeColor()}
              style={{ animation: "blink 4s infinite 0.2s" }}
            />
            <circle cx="126" cy="70" r="3" fill="white" />
          </g>

          {/* Eyebrows */}
          <g stroke="#8b7355" strokeWidth="2" fill="none" strokeLinecap="round">
            {avatarExpression === "thinking" ? (
              <>
                <path d="M 65 55 Q 75 50 85 55" />
                <path d="M 115 55 Q 125 50 135 55" />
              </>
            ) : avatarExpression === "sad" ? (
              <>
                <path d="M 65 55 Q 75 60 85 55" />
                <path d="M 115 55 Q 125 60 135 55" />
              </>
            ) : (
              <>
                <path d="M 65 55 Q 75 48 85 55" />
                <path d="M 115 55 Q 125 48 135 55" />
              </>
            )}
          </g>

          {/* Nose */}
          <path
            d="M 100 75 L 100 95 L 98 100 M 100 95 L 102 100"
            stroke="#d4a574"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
          />

          {/* Mouth */}
          <path
            d={getMouthPath()}
            stroke="#c85a54"
            strokeWidth="2"
            fill="none"
            strokeLinecap="round"
            style={{
              transition: isSpeaking ? "none" : "d 0.3s ease",
            }}
          />

          {/* Lips fill for speaking */}
          {isSpeaking && (
            <ellipse
              cx="100"
              cy={100 + mouthOpen * 8}
              rx="8"
              ry={3 + mouthOpen * 4}
              fill="#c85a54"
              opacity="0.3"
            />
          )}

          {/* Cheeks */}
          <circle cx="50" cy="85" r="12" fill="#ff6b6b" opacity="0.2" />
          <circle cx="150" cy="85" r="12" fill="#ff6b6b" opacity="0.2" />

          {/* Neck */}
          <rect
            x="85"
            y="140"
            width="30"
            height="25"
            fill="#fdbcb4"
            opacity="0.7"
          />

          {/* Thinking indicator */}
          {avatarExpression === "thinking" && (
            <g>
              <circle
                cx="140"
                cy="30"
                r="5"
                fill="#4f46e5"
                opacity="0.6"
                style={{
                  animation: "float 2s ease-in-out infinite",
                }}
              />
              <circle
                cx="155"
                cy="45"
                r="3"
                fill="#4f46e5"
                opacity="0.4"
                style={{
                  animation: "float 2.5s ease-in-out infinite 0.3s",
                }}
              />
            </g>
          )}

          {/* Listening indicator */}
          {isListening && (
            <g>
              <circle cx="100" cy="250" r="4" fill="#10b981" opacity="0.8">
                <animate
                  attributeName="r"
                  values="4;8;4"
                  dur="0.8s"
                  repeatCount="indefinite"
                />
              </circle>
              <circle
                cx="100"
                cy="250"
                r="10"
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
              >
                <animate
                  attributeName="r"
                  values="10;20;10"
                  dur="0.8s"
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="opacity"
                  values="1;0;1"
                  dur="0.8s"
                  repeatCount="indefinite"
                />
              </circle>
            </g>
          )}

          {/* Speaking indicator */}
          {isSpeaking && (
            <g>
              <path
                d="M 160 50 Q 175 70 160 90"
                stroke="#10b981"
                strokeWidth="2"
                fill="none"
                opacity="0.6"
              >
                <animate
                  attributeName="opacity"
                  values="0.6;0;0.6"
                  dur="0.6s"
                  repeatCount="indefinite"
                />
              </path>
              <path
                d="M 170 40 Q 190 70 170 100"
                stroke="#10b981"
                strokeWidth="2"
                fill="none"
                opacity="0.4"
              >
                <animate
                  attributeName="opacity"
                  values="0.4;0;0.4"
                  dur="0.8s"
                  repeatCount="indefinite"
                />
              </path>
            </g>
          )}
        </svg>

        <style>{`
          @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
          }
        `}</style>
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.header}>🎓 AI Virtual Tutor - Bob</h2>

      {/* Tab Navigation */}
      <div style={styles.tabContainer}>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === "chat" ? styles.activeTab : {}),
          }}
          onClick={() => handleTabChange("chat")}
        >
          💬 Chat
        </button>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === "profile" ? styles.activeTab : {}),
          }}
          onClick={() => handleTabChange("profile")}
        >
          👤 Your Profile
        </button>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === "history" ? styles.activeTab : {}),
          }}
          onClick={() => handleTabChange("history")}
        >
          📚 Learning History
        </button>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === "roadmap" ? styles.activeTab : {}),
          }}
          onClick={() => handleTabChange("roadmap")}
        >
          🗺️ Learning Roadmap
        </button>


      </div>

      {/* ROADMAP TAB */}
      {activeTab === "roadmap" && (
        <div style={styles.profileContainer}>
          <h3 style={styles.sectionTitle}>🗺️ Personalized Learning Roadmap</h3>

          <input
            style={styles.input}
            placeholder="What do you want to learn?"
            value={roadmapGoal}
            onChange={(e) => setRoadmapGoal(e.target.value)}
          />

          <select
            style={styles.input}
            value={studyTime}
            onChange={(e) => setStudyTime(e.target.value)}
          >
            <option value="0.5">30 minutes / day</option>
            <option value="1">1 hour / day</option>
            <option value="2">2 hours / day</option>
            <option value="3">3+ hours / day</option>
          </select>

          <select
            style={styles.input}
            value={learningStyle}
            onChange={(e) => setLearningStyle(e.target.value)}
          >
            <option value="visual">Visual</option>
            <option value="reading">Reading</option>
            <option value="kinesthetic">Hands-on</option>
            <option value="auditory">Auditory</option>
          </select>

          <button onClick={generateRoadmap} style={styles.button}>
            🚀 Generate Roadmap
          </button>

          {roadmapLoading && <p>⏳ Generating roadmap...</p>}

          {roadmapResult && roadmapResult.error && (
            <p style={{ color: "red" }}>{roadmapResult.error}</p>
          )}

          {roadmapResult && !roadmapResult.error && (
            <div style={styles.statCard}>
              <h4>📅 Weekly Plan</h4>
              <ul>
                {roadmapResult.weekly_plan.map((week, i) => (
                  <li key={i}>{week}</li>
                ))}
              </ul>

              <h4>🎯 Study Pattern</h4>
              <p>{roadmapResult.study_pattern}</p>

              <h4>📺 Recommended Resources</h4>
              <ul>
                {roadmapResult.resources.map((r, i) => (
                  <li key={i}>
                    <a href={r.link} target="_blank" rel="noreferrer">
                      {r.title}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}


      {/* Tab Content */}
      <div style={styles.contentArea}>
        {/* CHAT TAB with Realistic Avatar */}
        {activeTab === "chat" && (
          <div style={styles.chatLayout}>
            {/* Avatar Section */}
            <div style={styles.avatarSection}>
              <RealisticAvatar />

              {/* Voice Controls */}
              <div style={styles.voiceControls}>
                <button
                  onClick={isListening ? stopListening : startListening}
                  style={{
                    ...styles.voiceButton,
                    background: isListening ? "#ef4444" : "#10b981",
                  }}
                  disabled={loading || isSpeaking}
                >
                  {isListening ? "🎤 Listening..." : "🎤 Press to Speak"}
                </button>

                {isSpeaking && (
                  <button
                    onClick={stopSpeaking}
                    style={{
                      ...styles.voiceButton,
                      background: "#f59e0b",
                    }}
                  >
                    ⏹️ Stop Speaking
                  </button>
                )}
              </div>

              {/* Status */}
              <div style={styles.statusText}>
                {loading && "🤔 Bob is thinking..."}
                {isSpeaking && "🗣️ Bob is speaking..."}
                {isListening && "👂 Listening to you..."}
                {!loading && !isSpeaking && !isListening && "💬 Ready to help!"}
              </div>
            </div>

            {/* Chat Section */}
            <div style={styles.chatSection}>
              <div style={styles.chatBox} ref={chatBoxRef}>
                {messages.length === 0 && (
                  <div style={styles.welcomeMessage}>
                    <p>👋 Hi! I'm Bob, your AI tutor.</p>
                    <p>Click the microphone and start speaking!</p>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div
                    key={i}
                    style={{
                      ...styles.messageWrapper,
                      justifyContent:
                        msg.sender === "user" ? "flex-end" : "flex-start",
                    }}
                  >
                    <div
                      style={{
                        ...styles.message,
                        background:
                          msg.sender === "user" ? "#0078ff" : "#f1f1f1",
                        color: msg.sender === "user" ? "white" : "black",
                      }}
                    >
                      {msg.sender === "bot" && (
                        <button
                          onClick={() => speakText(msg.text)}
                          style={styles.speakButton}
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
                  <div style={styles.loadingWrapper}>
                    <div style={styles.loadingDots}>
                      <span style={styles.dot}>●</span>
                      <span style={styles.dot}>●</span>
                      <span style={styles.dot}>●</span>
                    </div>
                  </div>
                )}
              </div>

              <form onSubmit={sendQuestion} style={styles.inputArea}>
                <input
                  type="text"
                  placeholder="Or type your question..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  style={styles.input}
                  disabled={loading}
                />
                <button
                  type="submit"
                  style={{
                    ...styles.button,
                    opacity: loading ? 0.6 : 1,
                    cursor: loading ? "not-allowed" : "pointer",
                  }}
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
          <div style={styles.profileContainer}>
            {!studentProfile ? (
              <div style={styles.loadingCenter}>
                <p>Loading profile...</p>
              </div>
            ) : studentProfile.error ? (
              <div style={styles.errorBox}>
                <p>❌ {studentProfile.error}</p>
              </div>
            ) : (
              <>
                <h3 style={styles.sectionTitle}>📊 Your Learning Profile</h3>

                <div style={styles.statCard}>
                  <div style={styles.statLabel}>Total Interactions</div>
                  <div style={styles.statValue}>
                    {studentProfile.total_interactions || 0}
                  </div>
                </div>

                <div style={styles.statCard}>
                  <div style={styles.statLabel}>Learning Since</div>
                  <div style={styles.statValue}>
                    {studentProfile.first_interaction
                      ? new Date(
                          studentProfile.first_interaction
                        ).toLocaleDateString()
                      : "N/A"}
                  </div>
                </div>

                <h4 style={styles.subTitle}>📚 Topics You've Explored</h4>
                <div style={styles.topicsContainer}>
                  {studentProfile.topics_discussed &&
                  Object.keys(studentProfile.topics_discussed).length > 0 ? (
                    Object.entries(studentProfile.topics_discussed)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 10)
                      .map(([topic, count]) => (
                        <div key={topic} style={styles.topicTag}>
                          {topic}{" "}
                          <span style={styles.topicCount}>×{count}</span>
                        </div>
                      ))
                  ) : (
                    <p style={styles.emptyText}>
                      No topics yet. Start asking questions!
                    </p>
                  )}
                </div>

                <h4 style={styles.subTitle}>🎯 Question Types</h4>
                <div style={styles.questionTypes}>
                  {studentProfile.question_types &&
                  Object.keys(studentProfile.question_types).length > 0 ? (
                    Object.entries(studentProfile.question_types)
                      .sort((a, b) => b[1] - a[1])
                      .map(([type, count]) => (
                        <div key={type} style={styles.typeRow}>
                          <span style={styles.typeName}>
                            {type
                              .replace(/_/g, " ")
                              .replace(/\b\w/g, (l) => l.toUpperCase())}
                          </span>
                          <span style={styles.typeBar}>
                            <div
                              style={{
                                ...styles.typeBarFill,
                                width: `${(count / studentProfile.total_interactions) * 100}%`,
                              }}
                            />
                          </span>
                          <span style={styles.typeCount}>{count}</span>
                        </div>
                      ))
                  ) : (
                    <p style={styles.emptyText}>No data yet.</p>
                  )}
                </div>

                <button onClick={fetchProfile} style={styles.refreshButton}>
                  🔄 Refresh Profile
                </button>
              </>
            )}
          </div>
        )}

        {/* HISTORY TAB */}
        {activeTab === "history" && (
          <div style={styles.historyContainer}>
            <h3 style={styles.sectionTitle}>📚 Your Learning History</h3>
            <p style={styles.subtitle}>
              Recent conversations (showing last 20)
            </p>

            {interactions.length === 0 ? (
              <div style={styles.emptyText}>
                <p>No interactions yet. Start chatting with Bob!</p>
              </div>
            ) : (
              <div style={styles.interactionsList}>
                {interactions
                  .slice()
                  .reverse()
                  .map((interaction, idx) => (
                    <div key={idx} style={styles.interactionCard}>
                      <div style={styles.timestamp}>
                        🕒{" "}
                        {new Date(interaction.timestamp).toLocaleString()}
                      </div>
                      <div style={styles.interactionQ}>
                        <strong>Q:</strong> {interaction.question}
                      </div>
                      <div style={styles.interactionA}>
                        <strong>A:</strong> {interaction.answer}
                      </div>
                      {interaction.topics && interaction.topics.length > 0 && (
                        <div style={styles.interactionTopics}>
                          {interaction.topics.map((topic, i) => (
                            <span key={i} style={styles.miniTag}>
                              {topic}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            )}

            <button onClick={fetchInteractions} style={styles.refreshButton}>
              🔄 Refresh History
            </button>
          {/* ROADMAP TAB */}
          {activeTab === "roadmap" && (
            <div style={styles.profileContainer}>
              <h3 style={styles.sectionTitle}>🗺️ Personalized Learning Roadmap</h3>

              <input
                style={styles.input}
                placeholder="What do you want to learn?"
                value={roadmapGoal}
                onChange={(e) => setRoadmapGoal(e.target.value)}
              />

              <select
                style={styles.input}
                value={studyTime}
                onChange={(e) => setStudyTime(e.target.value)}
              >
                <option value="0.5">30 minutes / day</option>
                <option value="1">1 hour / day</option>
                <option value="2">2 hours / day</option>
              </select>

              <select
                style={styles.input}
                value={learningStyle}
                onChange={(e) => setLearningStyle(e.target.value)}
              >
                <option value="visual">Visual</option>
                <option value="reading">Reading</option>
                <option value="hands-on">Hands-on</option>
              </select>

              <button onClick={generateRoadmap} style={styles.button}>
                🚀 Generate Roadmap
              </button>

              {roadmapLoading && <p>Generating roadmap...</p>}

              {roadmapResult && !roadmapResult.error && (
                <div style={styles.statCard}>
                  <h4>📅 Weekly Plan</h4>
                  <ul>
                    {roadmapResult.weekly_plan.map((week, i) => (
                      <li key={i}>{week}</li>
                    ))}
                  </ul>

                  <h4>🎯 Study Pattern</h4>
                  <p>{roadmapResult.study_pattern}</p>

                  <h4>📺 Recommended Resources</h4>
                  <ul>
                    {roadmapResult.resources.map((r, i) => (
                      <li key={i}>
                        <a href={r.link} target="_blank" rel="noreferrer">
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
        )}
        
      </div>

      <p style={styles.footer}>
        
      </p>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    maxWidth: "1400px",
    margin: "20px auto",
    padding: "30px",
    borderRadius: "15px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
    backgroundColor: "white",
  },
  header: {
    textAlign: "center",
    marginBottom: "20px",
    color: "#333",
  },
  tabContainer: {
    display: "flex",
    gap: "5px",
    borderBottom: "2px solid #e0e0e0",
    marginBottom: "20px",
  },
  tab: {
    flex: 1,
    padding: "12px 20px",
    background: "transparent",
    border: "none",
    borderBottom: "3px solid transparent",
    cursor: "pointer",
    fontSize: "15px",
    fontWeight: "500",
    color: "#666",
    transition: "all 0.3s",
  },
  activeTab: {
    color: "#0078ff",
    borderBottomColor: "#0078ff",
    fontWeight: "600",
  },
  contentArea: {
    minHeight: "600px",
  },
  chatLayout: {
    display: "grid",
    gridTemplateColumns: "380px 1fr",
    gap: "30px",
    alignItems: "start",
  },
  avatarContainer: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100%",
  },
  avatarSection: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "20px",
    padding: "20px",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    borderRadius: "20px",
    boxShadow: "0 8px 30px rgba(0,0,0,0.12)",
  },
  voiceControls: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    width: "100%",
  },
  voiceButton: {
    padding: "12px 20px",
    border: "none",
    borderRadius: "12px",
    fontSize: "15px",
    fontWeight: "600",
    color: "white",
    cursor: "pointer",
    transition: "all 0.3s",
    boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
  },
  statusText: {
    color: "white",
    fontSize: "14px",
    fontWeight: "500",
    textAlign: "center",
    padding: "10px",
    background: "rgba(255,255,255,0.2)",
    borderRadius: "10px",
    minHeight: "40px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
  },
  chatSection: {
    display: "flex",
    flexDirection: "column",
    height: "600px",
  },
  chatBox: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    flex: 1,
    overflowY: "auto",
    border: "2px solid #e0e0e0",
    borderRadius: "12px",
    padding: "15px",
    marginBottom: "20px",
    backgroundColor: "#f9f9f9",
  },
  welcomeMessage: {
    textAlign: "center",
    color: "#999",
    marginTop: "200px",
    fontSize: "16px",
  },
  messageWrapper: {
    display: "flex",
    width: "100%",
  },
  message: {
    padding: "12px 18px",
    borderRadius: "18px",
    maxWidth: "75%",
    wordWrap: "break-word",
    fontSize: "15px",
    lineHeight: "1.5",
    boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
    position: "relative",
  },
  speakButton: {
    position: "absolute",
    top: "5px",
    right: "5px",
    background: "transparent",
    border: "none",
    cursor: "pointer",
    fontSize: "16px",
    opacity: 0.7,
    transition: "opacity 0.2s",
  },
  loadingWrapper: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "10px",
  },
  loadingDots: {
    display: "flex",
    gap: "4px",
  },
  dot: {
    fontSize: "20px",
    color: "#0078ff",
    animation: "bounce 1.4s infinite ease-in-out",
  },
  inputArea: {
    display: "flex",
    gap: "12px",
  },
  input: {
    flex: 1,
    padding: "14px",
    borderRadius: "10px",
    border: "2px solid #e0e0e0",
    fontSize: "15px",
    outline: "none",
  },
  button: {
    padding: "14px 30px",
    background: "#0078ff",
    color: "white",
    border: "none",
    borderRadius: "10px",
    fontSize: "15px",
    fontWeight: "600",
    cursor: "pointer",
  },
  profileContainer: {
    padding: "20px",
    overflowY: "auto",
    maxHeight: "600px",
  },
  sectionTitle: {
    color: "#333",
    marginBottom: "20px",
  },
  statCard: {
    background: "#f8f9fa",
    padding: "15px",
    borderRadius: "10px",
    marginBottom: "15px",
    border: "1px solid #e0e0e0",
  },
  statLabel: {
    fontSize: "13px",
    color: "#666",
    marginBottom: "5px",
  },
  statValue: {
    fontSize: "24px",
    fontWeight: "bold",
    color: "#0078ff",
  },
  subTitle: {
    color: "#555",
    marginTop: "25px",
    marginBottom: "15px",
  },
  topicsContainer: {
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
    marginBottom: "20px",
  },
  topicTag: {
    background: "#e3f2fd",
    color: "#0078ff",
    padding: "8px 15px",
    borderRadius: "20px",
    fontSize: "14px",
    fontWeight: "500",
  },
  topicCount: {
    fontWeight: "bold",
    marginLeft: "5px",
  },
  questionTypes: {
    marginBottom: "20px",
  },
  typeRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    marginBottom: "10px",
  },
  typeName: {
    flex: "0 0 180px",
    fontSize: "14px",
    color: "#555",
  },
  typeBar: {
    flex: 1,
    height: "20px",
    background: "#e0e0e0",
    borderRadius: "10px",
    overflow: "hidden",
  },
  typeBarFill: {
    height: "100%",
    background: "linear-gradient(90deg, #0078ff, #00b4ff)",
    transition: "width 0.3s",
  },
  typeCount: {
    flex: "0 0 40px",
    textAlign: "right",
    fontWeight: "bold",
    color: "#0078ff",
  },
  historyContainer: {
    padding: "20px",
    overflowY: "auto",
    maxHeight: "600px",
  },
  subtitle: {
    color: "#666",
    fontSize: "14px",
    marginBottom: "20px",
  },
  interactionsList: {
    display: "flex",
    flexDirection: "column",
    gap: "15px",
  },
  interactionCard: {
    background: "#f8f9fa",
    padding: "15px",
    borderRadius: "10px",
    border: "1px solid #e0e0e0",
  },
  timestamp: {
    fontSize: "12px",
    color: "#999",
    marginBottom: "10px",
  },
  interactionQ: {
    marginBottom: "10px",
    fontSize: "14px",
    color: "#333",
  },
  interactionA: {
    fontSize: "14px",
    color: "#555",
    marginBottom: "10px",
    paddingLeft: "10px",
    borderLeft: "3px solid #0078ff",
  },
  interactionTopics: {
    display: "flex",
    gap: "5px",
    flexWrap: "wrap",
    marginTop: "10px",
  },
  miniTag: {
    background: "#e3f2fd",
    color: "#0078ff",
    padding: "4px 10px",
    borderRadius: "12px",
    fontSize: "12px",
  },
  refreshButton: {
    marginTop: "20px",
    padding: "10px 20px",
    background: "#0078ff",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: "600",
  },
  emptyText: {
    textAlign: "center",
    color: "#999",
    padding: "40px 20px",
    fontSize: "15px",
  },
  loadingCenter: {
    textAlign: "center",
    padding: "40px",
    color: "#666",
  },
  errorBox: {
    background: "#ffebee",
    color: "#c62828",
    padding: "20px",
    borderRadius: "10px",
    textAlign: "center",
  },
  footer: {
    textAlign: "center",
    fontSize: "12px",
    color: "#999",
    marginTop: "20px",
  },
};

export default App;