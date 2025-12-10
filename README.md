# Docusaurus Project Structure for Physical AI & Humanoid Robotics Book

This project contains the "Physical AI & Humanoid Robotics: From Simulated Brains to Walking Bodies" textbook, implemented as a Docusaurus site.

## Project Structure

```
/mnt/d/Quarter-4/spec_kit_plus/humenoid_robot/
├── docs/                    # Book content organized by parts/chapters
├── src/
│   ├── components/          # Custom React components for robotics content
│   ├── css/                # Custom styles with robotics theme
│   └── pages/              # Additional pages
├── static/                 # Static assets (images, code samples)
├── docusaurus.config.js    # Main configuration
├── package.json            # Dependencies
├── sidebars.js             # Navigation structure
└── README.md               # Project overview
```

## Book Architecture (18 Chapters across 6 Parts)

### Part I: Foundations & Nervous System (ROS 2)
- Chapter 1: Introduction to Physical AI and Humanoid Robotics (10,000 words)
- Chapter 2: ROS 2 Fundamentals for Humanoid Systems (11,000 words)
- Chapter 3: URDF and Robot Modeling (10,000 words)

### Part II: Digital Twins & Simulation Mastery
- Chapter 4: Isaac Sim Fundamentals and Scene Creation (12,000 words)
- Chapter 5: Advanced Simulation Techniques (11,000 words)
- Chapter 6: Simulation-to-Reality Transfer (10,000 words)

### Part III: Perception & Edge Brain
- Chapter 7: Isaac ROS Perception Pipeline (11,000 words)
- Chapter 8: Vision-Language-Action Models for Humanoids (12,000 words)
- Chapter 9: Edge Computing for Real-time Perception (10,000 words)

### Part IV: Embodied Cognition & VLA Models
- Chapter 10: Cognitive Architectures for Humanoid Robots (11,000 words)
- Chapter 11: Large Language Models Integration (10,000 words)
- Chapter 12: Vision-Language Integration (10,000 words)

### Part V: Bipedal Locomotion & Whole-Body Control
- Chapter 13: Introduction to Bipedal Locomotion (11,000 words)
- Chapter 14: Whole-Body Control Strategies (12,000 words)
- Chapter 15: Adaptive and Learning-Based Control (11,000 words)

### Part VI: Capstone Integration & Sim-to-Real Transfer
- Chapter 16: System Integration and Architecture (10,000 words)
- Chapter 17: Capstone Project Implementation (12,000 words)
- Chapter 18: Deployment and Real-World Operation (10,000 words)

## Technology Stack
- Docusaurus for static site generation
- React components for interactive content
- Custom CSS for robotics-themed styling
- GitHub Pages for deployment