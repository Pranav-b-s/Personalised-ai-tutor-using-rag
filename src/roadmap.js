// Roadmap Page JavaScript
import React, { useState } from 'react';
import axios from 'axios';
import './Roadmap.css';

document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generateBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const roadmapGoalInput = document.getElementById('roadmapGoal');
    const studyTimeSelect = document.getElementById('studyTime');
    const learningStyleSelect = document.getElementById('learningStyle');
    
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    const roadmapResult = document.getElementById('roadmapResult');
    
    let currentRoadmap = null;
  
    // Generate Roadmap
    generateBtn.addEventListener('click', async function() {
      const goal = roadmapGoalInput.value.trim();
      const timePerDay = studyTimeSelect.value;
      const learningStyle = learningStyleSelect.value;
  
      if (!goal) {
        showError('Please enter a learning goal');
        return;
      }
  
      // Hide previous results
      roadmapResult.style.display = 'none';
      errorState.style.display = 'none';
      
      // Show loading
      loadingState.style.display = 'block';
  
      try {
        const response = await fetch('http://127.0.0.1:5000/learning-roadmap', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            goal: goal,
            time_per_day: timePerDay,
            learning_style: learningStyle
          })
        });
  
        const data = await response.json();
  
        if (data.error) {
          showError(data.error);
          return;
        }
  
        currentRoadmap = data;
        displayRoadmap(data);
        
      } catch (error) {
        console.error('Error:', error);
        showError('Failed to generate roadmap. Make sure the Flask backend is running on port 5000.');
      } finally {
        loadingState.style.display = 'none';
      }
    });
  
    // Download PDF (simplified - just print for now)
    downloadBtn.addEventListener('click', function() {
      window.print();
    });
  
    function showError(message) {
      errorState.style.display = 'block';
      document.getElementById('errorMessage').textContent = `❌ ${message}`;
      loadingState.style.display = 'none';
      roadmapResult.style.display = 'none';
    }
  
    function displayRoadmap(data) {
      // Study Commitment
      document.getElementById('studyCommitment').textContent = 
        `${data.time_per_day} ${data.time_per_day == 1 ? 'hour' : 'hours'} per day for ${data.weeks} weeks`;
  
      // Weekly Plan
      const weeklyPlanList = document.getElementById('weeklyPlan');
      weeklyPlanList.innerHTML = '';
      data.weekly_plan.forEach(week => {
        const li = document.createElement('li');
        li.textContent = week;
        weeklyPlanList.appendChild(li);
      });
  
      // Study Pattern
      document.getElementById('studyPattern').textContent = data.study_pattern;
  
      // Resources
      const resourcesGrid = document.getElementById('resources');
      resourcesGrid.innerHTML = '';
      data.resources.forEach(resource => {
        const card = document.createElement('a');
        card.className = 'resource-card';
        card.href = resource.link;
        card.target = '_blank';
        card.rel = 'noopener noreferrer';
        
        const title = document.createElement('h5');
        title.textContent = resource.title;
        
        const description = document.createElement('p');
        description.textContent = 'Click to explore →';
        
        card.appendChild(title);
        card.appendChild(description);
        resourcesGrid.appendChild(card);
      });
  
      // Show result
      roadmapResult.style.display = 'block';
      
      // Smooth scroll to result
      setTimeout(() => {
        roadmapResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  
    // Enter key support
    roadmapGoalInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        generateBtn.click();
      }
    });
  });