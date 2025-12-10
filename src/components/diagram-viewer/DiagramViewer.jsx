// src/components/diagram-viewer/DiagramViewer.jsx
// Interactive diagram viewer for robotics illustrations
import React, { useState, useRef, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './DiagramViewer.module.css';

const DiagramViewer = ({
  src,
  alt = "Robotics Diagram",
  title = "Robotics Diagram Viewer",
  description = "Interactive viewer for robotics diagrams and technical illustrations",
  width = "100%",
  height = "500px",
  showControls = true,
  enableZoom = true,
  enablePan = true
}) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5));
  };

  const handleReset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e) => {
    if (!enablePan) return;
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !enablePan) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    if (!enableZoom) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setScale(prev => Math.max(0.5, Math.min(prev + delta, 3)));
  };

  // Reset drag state when component unmounts
  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  return (
    <div className={styles.diagramViewer}>
      <div className={styles.viewerHeader}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      {showControls && (
        <div className={styles.viewerControls}>
          <button className={styles.controlButton} onClick={handleZoomIn} title="Zoom In">
            +
          </button>
          <button className={styles.controlButton} onClick={handleZoomOut} title="Zoom Out">
            -
          </button>
          <button className={styles.controlButton} onClick={handleReset} title="Reset View">
            ↻
          </button>
          <span className={styles.zoomLevel}>Zoom: {Math.round(scale * 100)}%</span>
        </div>
      )}

      <div
        ref={containerRef}
        className={styles.viewerContainer}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        style={{ width, height }}
      >
        <img
          ref={imageRef}
          src={src}
          alt={alt}
          className={styles.diagramImage}
          style={{
            transform: `scale(${scale}) translate(${position.x}px, ${position.y}px)`,
            cursor: isDragging ? 'grabbing' : enablePan ? 'grab' : 'default',
            transition: isDragging ? 'none' : 'transform 0.1s ease'
          }}
        />
        {isDragging && enablePan && (
          <div className={styles.dragIndicator}>
            Dragging...
          </div>
        )}
      </div>

      <div className={styles.viewerInfo}>
        <p>Click and drag to pan the diagram. Use the controls to zoom in/out or reset the view.</p>
      </div>
    </div>
  );
};

// Wrapper component to handle client-side rendering
const DiagramViewerWrapper = (props) => {
  return (
    <BrowserOnly>
      {() => <DiagramViewer {...props} />}
    </BrowserOnly>
  );
};

export default DiagramViewerWrapper;