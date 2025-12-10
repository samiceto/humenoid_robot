# Interactive Diagram Viewers

This section provides interactive viewers for robotics diagrams and technical illustrations. These viewers allow you to zoom, pan, and explore complex robotics diagrams in detail.

## Sample Humanoid Robot Diagram

The following diagram shows a typical humanoid robot structure with key components labeled:

import HumanoidDiagramViewer from '@site/src/components/diagram-viewer/HumanoidDiagramViewer';

<HumanoidDiagramViewer
  src="img/humanoid-robot-structure.svg"
  title="Humanoid Robot Structure"
  description="Technical diagram showing the kinematic structure of a humanoid robot with 20+ degrees of freedom"
/>

## Basic Diagram Viewer

For general robotics diagrams, you can use the basic diagram viewer:

import DiagramViewer from '@site/src/components/diagram-viewer/DiagramViewer';

<DiagramViewer
  src="img/humanoid-robot-structure.svg"
  title="ROS 2 Architecture"
  description="Architecture diagram showing the ROS 2 middleware and communication patterns"
  width="100%"
  height="400px"
/>

## URDF Model Visualization

The diagram viewer can also display URDF model visualizations:

<DiagramViewer
  src="img/humanoid-robot-structure.svg"
  title="TurtleBot 4 URDF Model"
  description="URDF representation of the TurtleBot 4 mobile robot platform"
  width="100%"
  height="500px"
/>

## How to Use the Diagram Viewers

1. **Zoom**: Use the + and - buttons or mouse wheel to zoom in and out
2. **Pan**: Click and drag to move around the diagram
3. **Reset**: Use the reset button to return to the original view
4. **Responsive**: Diagrams automatically adjust to different screen sizes

## Features

- **Zoom**: Up to 3x magnification for detailed inspection
- **Pan**: Smooth panning across large diagrams
- **Responsive**: Works on desktop and mobile devices
- **Accessible**: Keyboard navigable and screen reader friendly
- **High Contrast**: Supports high contrast mode for better visibility

## Available Diagram Types

The diagram viewers support various types of robotics illustrations:

- Kinematic structures and joint configurations
- URDF/SDF model visualizations
- ROS 2 architecture diagrams
- Control system block diagrams
- Sensor placement and coverage areas
- Motion planning and navigation diagrams
- Hardware component layouts

## Accessibility Features

The diagram viewers include several accessibility features:

- Keyboard navigation support
- High contrast mode compatibility
- Screen reader friendly labels
- Focus indicators for interactive elements
- Reduced motion options for motion sensitivity

For the best experience with large diagrams, we recommend using a desktop computer with a mouse for precise zooming and panning.