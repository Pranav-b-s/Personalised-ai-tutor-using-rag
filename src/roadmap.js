import React, { useState, useEffect } from 'react';

function Roadmap() {
  const [topic, setTopic] = useState('');
  const [timeline, setTimeline] = useState('week');
  const [duration, setDuration] = useState(4);
  const [roadmap, setRoadmap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modifying, setModifying] = useState(false);
  const [modificationRequest, setModificationRequest] = useState('');
  const [acceptedRoadmaps, setAcceptedRoadmaps] = useState([]);
  const [showProgressPopup, setShowProgressPopup] = useState(false);
  const [currentTask, setCurrentTask] = useState(null);
  const [viewingRoadmap, setViewingRoadmap] = useState(null);

  useEffect(() => {
    loadAcceptedRoadmaps();
    checkTaskProgress();
    
    // Check progress every hour
    const interval = setInterval(checkTaskProgress, 3600000);
    return () => clearInterval(interval);
  }, []);

  const loadAcceptedRoadmaps = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/accepted-roadmaps');
      const data = await response.json();
      setAcceptedRoadmaps(data.roadmaps || []);
    } catch (error) {
      console.error('Error loading roadmaps:', error);
    }
  };

  const checkTaskProgress = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/check-progress');
      const data = await response.json();
      if (data.task_due) {
        setCurrentTask(data.task);
        setShowProgressPopup(true);
      }
    } catch (error) {
      console.error('Error checking progress:', error);
    }
  };

  const generateRoadmap = async () => {
    if (!topic.trim()) {
      alert('Please enter a topic to study');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/generate-roadmap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, timeline, duration })
      });
      const data = await response.json();
      setRoadmap(data.roadmap);
    } catch (error) {
      console.error('Error generating roadmap:', error);
      alert('Failed to generate roadmap. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const modifyRoadmap = async () => {
    if (!modificationRequest.trim()) {
      alert('Please enter what you want to change');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/modify-roadmap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roadmap, modification_request: modificationRequest })
      });
      const data = await response.json();
      setRoadmap(data.roadmap);
      setModifying(false);
      setModificationRequest('');
    } catch (error) {
      console.error('Error modifying roadmap:', error);
      alert('Failed to modify roadmap. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const acceptRoadmap = async () => {
    setLoading(true);
    try {
      await fetch('http://127.0.0.1:5000/accept-roadmap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roadmap })
      });
      alert('Roadmap accepted and saved successfully!');
      loadAcceptedRoadmaps();
      setRoadmap(null);
      setTopic('');
    } catch (error) {
      console.error('Error accepting roadmap:', error);
      alert('Failed to save roadmap. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleProgressResponse = async (completed) => {
    try {
      await fetch('http://127.0.0.1:5000/update-progress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: currentTask, completed })
      });
      setShowProgressPopup(false);
      setCurrentTask(null);
      
      if (!completed) {
        alert('Roadmap will be adjusted based on your progress.');
        loadAcceptedRoadmaps();
      }
    } catch (error) {
      console.error('Error updating progress:', error);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '2rem',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        background: 'white',
        borderRadius: '16px',
        padding: '2rem',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
      }}>
        <h1 style={{
          textAlign: 'center',
          color: '#667eea',
          marginBottom: '2rem',
          fontSize: '2.5rem'
        }}>
          🗺️ AI Study Roadmap Generator
        </h1>

        {/* Input Section */}
        {!roadmap && !viewingRoadmap && (
          <div style={{
            background: '#f8f9fa',
            padding: '2rem',
            borderRadius: '12px',
            marginBottom: '2rem'
          }}>
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{
                display: 'block',
                marginBottom: '0.5rem',
                fontWeight: 'bold',
                color: '#333'
              }}>
                Topic to Study
              </label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Machine Learning, React.js, Data Structures"
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '2px solid #ddd',
                  borderRadius: '8px',
                  fontSize: '1rem'
                }}
              />
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '1rem',
              marginBottom: '1.5rem'
            }}>
              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '0.5rem',
                  fontWeight: 'bold',
                  color: '#333'
                }}>
                  Timeline Unit
                </label>
                <select
                  value={timeline}
                  onChange={(e) => setTimeline(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px solid #ddd',
                    borderRadius: '8px',
                    fontSize: '1rem'
                  }}
                >
                  <option value="day">Days</option>
                  <option value="week">Weeks</option>
                  <option value="month">Months</option>
                </select>
              </div>

              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '0.5rem',
                  fontWeight: 'bold',
                  color: '#333'
                }}>
                  Duration
                </label>
                <input
                  type="number"
                  min="1"
                  max="52"
                  value={duration}
                  onChange={(e) => setDuration(parseInt(e.target.value))}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '2px solid #ddd',
                    borderRadius: '8px',
                    fontSize: '1rem'
                  }}
                />
              </div>
            </div>

            <button
              onClick={generateRoadmap}
              disabled={loading}
              style={{
                width: '100%',
                padding: '1rem',
                background: loading ? '#ccc' : '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1.1rem',
                fontWeight: 'bold',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s'
              }}
            >
              {loading ? '🔄 Generating...' : '✨ Generate Roadmap'}
            </button>
          </div>
        )}

        {/* Viewing Roadmap Details */}
        {viewingRoadmap && (
          <div style={{
            background: '#f8f9fa',
            padding: '2rem',
            borderRadius: '12px',
            marginBottom: '2rem'
          }}>
            <button
              onClick={() => setViewingRoadmap(null)}
              style={{
                padding: '0.5rem 1rem',
                background: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '0.9rem',
                fontWeight: 'bold',
                cursor: 'pointer',
                marginBottom: '1rem'
              }}
            >
              ← Back to List
            </button>

            <h2 style={{ color: '#333', marginBottom: '1rem' }}>
              📚 {viewingRoadmap.topic}
            </h2>
            <p style={{ color: '#666', marginBottom: '1.5rem' }}>
              Duration: {viewingRoadmap.duration} {viewingRoadmap.timeline}s | Progress: {viewingRoadmap.completed_tasks || 0} / {viewingRoadmap.total_tasks} tasks
            </p>

            {viewingRoadmap.phases && viewingRoadmap.phases.map((phase, phaseIdx) => (
              <div key={phaseIdx} style={{
                background: 'white',
                padding: '1.5rem',
                borderRadius: '8px',
                marginBottom: '1rem',
                border: '2px solid #e0e0e0'
              }}>
                <h3 style={{ color: '#667eea', marginBottom: '0.5rem' }}>
                  Phase {phaseIdx + 1}: {phase.name}
                </h3>
                <p style={{ color: '#888', fontSize: '0.9rem', marginBottom: '1rem' }}>
                  {phase.timeline}
                </p>

                {phase.tasks && phase.tasks.map((task, taskIdx) => (
                  <div key={taskIdx} style={{
                    padding: '0.75rem',
                    background: task.completed ? '#d4edda' : '#f8f9fa',
                    borderRadius: '6px',
                    marginBottom: '0.5rem',
                    borderLeft: `4px solid ${task.completed ? '#28a745' : '#667eea'}`
                  }}>
                    <div style={{ fontWeight: 'bold', color: '#333' }}>
                      {task.completed && '✅ '}{task.name}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.25rem' }}>
                      {task.description}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.25rem' }}>
                      ⏱️ {task.estimated_time}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        {/* Roadmap Display */}
        {roadmap && !viewingRoadmap && (
          <div style={{
            background: '#f8f9fa',
            padding: '2rem',
            borderRadius: '12px',
            marginBottom: '2rem'
          }}>
            <h2 style={{ color: '#333', marginBottom: '1rem' }}>
              📚 {roadmap.topic}
            </h2>
            <p style={{ color: '#666', marginBottom: '1.5rem' }}>
              Duration: {roadmap.duration} {roadmap.timeline}s | Total Tasks: {roadmap.phases.reduce((acc, phase) => acc + phase.tasks.length, 0)}
            </p>

            {roadmap.phases.map((phase, phaseIdx) => (
              <div key={phaseIdx} style={{
                background: 'white',
                padding: '1.5rem',
                borderRadius: '8px',
                marginBottom: '1rem',
                border: '2px solid #e0e0e0'
              }}>
                <h3 style={{ color: '#667eea', marginBottom: '0.5rem' }}>
                  Phase {phaseIdx + 1}: {phase.name}
                </h3>
                <p style={{ color: '#888', fontSize: '0.9rem', marginBottom: '1rem' }}>
                  {phase.timeline}
                </p>

                {phase.tasks.map((task, taskIdx) => (
                  <div key={taskIdx} style={{
                    padding: '0.75rem',
                    background: '#f8f9fa',
                    borderRadius: '6px',
                    marginBottom: '0.5rem',
                    borderLeft: '4px solid #667eea'
                  }}>
                    <div style={{ fontWeight: 'bold', color: '#333' }}>
                      {task.name}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.25rem' }}>
                      {task.description}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.25rem' }}>
                      ⏱️ {task.estimated_time}
                    </div>
                  </div>
                ))}
              </div>
            ))}

            {/* Action Buttons */}
            {!modifying ? (
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '1rem',
                marginTop: '1.5rem'
              }}>
                <button
                  onClick={() => setModifying(true)}
                  style={{
                    padding: '1rem',
                    background: '#ffc107',
                    color: '#333',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    cursor: 'pointer'
                  }}
                >
                  ✏️ Modify
                </button>
                <button
                  onClick={acceptRoadmap}
                  disabled={loading}
                  style={{
                    padding: '1rem',
                    background: loading ? '#ccc' : '#28a745',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    cursor: loading ? 'not-allowed' : 'pointer'
                  }}
                >
                  ✅ Accept & Save
                </button>
              </div>
            ) : (
              <div style={{ marginTop: '1.5rem' }}>
                <textarea
                  value={modificationRequest}
                  onChange={(e) => setModificationRequest(e.target.value)}
                  placeholder="What would you like to change? (e.g., 'Add more hands-on projects', 'Reduce duration of Phase 2')"
                  style={{
                    width: '100%',
                    minHeight: '100px',
                    padding: '1rem',
                    border: '2px solid #ddd',
                    borderRadius: '8px',
                    fontSize: '1rem',
                    marginBottom: '1rem',
                    resize: 'vertical'
                  }}
                />
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '1rem'
                }}>
                  <button
                    onClick={() => {
                      setModifying(false);
                      setModificationRequest('');
                    }}
                    style={{
                      padding: '1rem',
                      background: '#dc3545',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      fontSize: '1rem',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}
                  >
                    ❌ Cancel
                  </button>
                  <button
                    onClick={modifyRoadmap}
                    disabled={loading}
                    style={{
                      padding: '1rem',
                      background: loading ? '#ccc' : '#667eea',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      fontSize: '1rem',
                      fontWeight: 'bold',
                      cursor: loading ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {loading ? '🔄 Modifying...' : '🔄 Apply Changes'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Accepted Roadmaps */}
        {acceptedRoadmaps.length > 0 && !roadmap && !viewingRoadmap && (
          <div>
            <h2 style={{ color: '#333', marginBottom: '1rem' }}>
              📋 Your Active Roadmaps
            </h2>
            {acceptedRoadmaps.map((rm, idx) => (
              <div key={idx} style={{
                background: '#f8f9fa',
                padding: '1.5rem',
                borderRadius: '12px',
                marginBottom: '1rem',
                border: '2px solid #667eea'
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: '0.5rem'
                }}>
                  <h3 style={{ color: '#667eea', margin: 0 }}>
                    {rm.topic}
                  </h3>
                  <button
                    onClick={() => setViewingRoadmap(rm)}
                    style={{
                      padding: '0.5rem 1rem',
                      background: '#667eea',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      fontSize: '0.9rem',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      whiteSpace: 'nowrap'
                    }}
                  >
                    👁️ View
                  </button>
                </div>
                <div style={{ fontSize: '0.9rem', color: '#666' }}>
                  Progress: {rm.completed_tasks || 0} / {rm.total_tasks} tasks completed
                </div>
                <div style={{
                  marginTop: '0.5rem',
                  height: '8px',
                  background: '#e0e0e0',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${((rm.completed_tasks || 0) / rm.total_tasks) * 100}%`,
                    height: '100%',
                    background: '#28a745',
                    transition: 'width 0.3s'
                  }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Progress Check Popup */}
      {showProgressPopup && currentTask && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            padding: '2rem',
            borderRadius: '16px',
            maxWidth: '500px',
            width: '90%'
          }}>
            <h2 style={{ color: '#667eea', marginBottom: '1rem' }}>
              ⏰ Task Progress Check
            </h2>
            <p style={{ color: '#333', marginBottom: '1rem' }}>
              Have you completed the following task?
            </p>
            <div style={{
              background: '#f8f9fa',
              padding: '1rem',
              borderRadius: '8px',
              marginBottom: '1.5rem'
            }}>
              <strong>{currentTask.name}</strong>
              <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.5rem' }}>
                {currentTask.description}
              </p>
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '1rem'
            }}>
              <button
                onClick={() => handleProgressResponse(false)}
                style={{
                  padding: '1rem',
                  background: '#dc3545',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  cursor: 'pointer'
                }}
              >
                ❌ Not Yet
              </button>
              <button
                onClick={() => handleProgressResponse(true)}
                style={{
                  padding: '1rem',
                  background: '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  cursor: 'pointer'
                }}
              >
                ✅ Completed
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Roadmap;