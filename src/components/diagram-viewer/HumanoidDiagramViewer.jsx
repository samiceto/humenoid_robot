// src/components/diagram-viewer/HumanoidDiagramViewer.jsx
// Specialized diagram viewer for humanoid robotics illustrations
import React from 'react';
import DiagramViewer from './DiagramViewer';
import styles from './HumanoidDiagramViewer.module.css';

const HumanoidDiagramViewer = ({
  src,
  title = "Humanoid Robot Diagram",
  description = "Technical diagram of humanoid robot kinematics and structure",
  ...props
}) => {
  return (
    <div className={styles.humanoidDiagramContainer}>
      <DiagramViewer
        src={src}
        title={title}
        description={description}
        width="100%"
        height="600px"
        showControls={true}
        enableZoom={true}
        enablePan={true}
        {...props}
      />
      <div className={styles.diagramLegend}>
        <h4>Diagram Legend</h4>
        <ul>
          <li><span className={styles.jointMarker}>●</span> Joint</li>
          <li><span className={styles.linkMarker}>▬</span> Link/Segment</li>
          <li><span className={styles.actuatorMarker}>▲</span> Actuator/Motor</li>
          <li><span className={styles.sensorMarker}>◆</span> Sensor</li>
          <li><span className={styles.cofMarker}>■</span> Center of Force</li>
        </ul>
      </div>
    </div>
  );
};

export default HumanoidDiagramViewer;