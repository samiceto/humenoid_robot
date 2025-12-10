// src/components/simulation-preview/SimulationPreview.jsx
// Interactive simulation preview for Isaac Sim content
import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './SimulationPreview.module.css';

const SimulationPreview = ({
  title = "Isaac Sim Preview",
  description = "Interactive preview of Isaac Sim simulation environment",
  width = "100%",
  height = "500px",
  showControls = true,
  defaultScene = "basic_cubicle",
  showSceneSelector = true
}) => {
  const [currentScene, setCurrentScene] = useState(defaultScene);
  const [isPlaying, setIsPlaying] = useState(false);
  const [simulationTime, setSimulationTime] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);

  // Simulated scenes for Isaac Sim
  const scenes = [
    { id: 'basic_cubicle', name: 'Basic Cubicle', description: 'Simple environment with basic shapes' },
    { id: 'warehouse', name: 'Warehouse', description: 'Industrial warehouse environment' },
    { id: 'hospital', name: 'Hospital', description: 'Hospital corridor environment' },
    { id: 'home', name: 'Home', description: 'Residential living space' },
    { id: 'outdoor', name: 'Outdoor', description: 'Outdoor terrain with obstacles' }
  ];

  // Simulate loading of the simulation environment
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 1000);
    return () => clearTimeout(timer);
  }, [currentScene]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setSimulationTime(0);
    setIsPlaying(false);
  };

  const handleSceneChange = (sceneId) => {
    setIsLoaded(false);
    setCurrentScene(sceneId);
  };

  // Simulate time progression when playing
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setSimulationTime(time => time + 0.1);
      }, 100);
    } else if (!isPlaying) {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  return (
    <div className={styles.simulationPreview}>
      <div className={styles.previewHeader}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      {showSceneSelector && (
        <div className={styles.sceneSelector}>
          <label htmlFor="scene-select">Select Scene: </label>
          <select
            id="scene-select"
            value={currentScene}
            onChange={(e) => handleSceneChange(e.target.value)}
            className={styles.sceneSelect}
          >
            {scenes.map(scene => (
              <option key={scene.id} value={scene.id}>
                {scene.name}
              </option>
            ))}
          </select>
          <span className={styles.sceneDescription}>
            {scenes.find(s => s.id === currentScene)?.description}
          </span>
        </div>
      )}

      {showControls && (
        <div className={styles.previewControls}>
          <button
            className={`${styles.controlButton} ${isPlaying ? styles.playing : ''}`}
            onClick={handlePlayPause}
            title={isPlaying ? "Pause Simulation" : "Play Simulation"}
          >
            {isPlaying ? '⏸' : '▶'} {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            className={styles.controlButton}
            onClick={handleReset}
            title="Reset Simulation"
          >
            ⏮ Reset
          </button>
          <span className={styles.simulationTime}>
            Time: {simulationTime.toFixed(1)}s
          </span>
        </div>
      )}

      <div
        className={styles.simulationContainer}
        style={{ width, height }}
      >
        {!isLoaded ? (
          <div className={styles.loadingOverlay}>
            <div className={styles.loadingSpinner}>Loading simulation...</div>
          </div>
        ) : (
          <div className={styles.simulationContent}>
            <div className={styles.simulationPlaceholder}>
              <div className={styles.simulationScene}>
                {/* Placeholder for Isaac Sim preview */}
                <div className={styles.robotModel}>
                  <div className={styles.robotBody}></div>
                  <div className={styles.robotHead}></div>
                  <div className={styles.robotArm}></div>
                  <div className={styles.robotArm} style={{transform: 'scaleX(-1)'}}></div>
                  <div className={styles.robotLeg}></div>
                  <div className={styles.robotLeg} style={{left: 'calc(50% + 20px)'}}></div>
                </div>

                <div className={styles.environment}>
                  <div className={styles.floor}></div>
                  <div className={styles.obstacle}></div>
                  <div className={styles.target}></div>
                </div>
              </div>

              <div className={styles.simulationInfo}>
                <h4>Isaac Sim Environment: {scenes.find(s => s.id === currentScene)?.name}</h4>
                <p>Real-time physics simulation with NVIDIA PhysX</p>
                <p>ROS 2 bridge enabled for robot control</p>
                <p>Sensor data: RGB, Depth, IMU, LIDAR</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className={styles.previewInfo}>
        <p>This is a simulation preview. In a real Isaac Sim environment, you would see:</p>
        <ul>
          <li>Real-time physics simulation with NVIDIA PhysX</li>
          <li>High-fidelity graphics rendering</li>
          <li>Accurate sensor simulation (cameras, LiDAR, IMU)</li>
          <li>ROS 2 integration for robot control</li>
          <li>Domain randomization capabilities</li>
        </ul>
      </div>
    </div>
  );
};

// Wrapper component to handle client-side rendering
const SimulationPreviewWrapper = (props) => {
  return (
    <BrowserOnly>
      {() => <SimulationPreview {...props} />}
    </BrowserOnly>
  );
};

export default SimulationPreviewWrapper;