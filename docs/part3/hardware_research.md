# Hardware Research and Lab Setup Options

## Executive Summary

This document provides comprehensive research and recommendations for lab setup options for the Physical AI & Humanoid Robotics course, offering three distinct tiers to accommodate various budget constraints while ensuring students can achieve the learning outcomes.

## Lab Tier Recommendations

### Budget Tier (&lt;$1k) - "Simulation-First" Approach

**Target Audience**: Students with limited hardware budget, focusing on simulation-based learning

**Core Components**:
- **Computer**: Mid-range laptop/desktop with Ubuntu 22.04
  - CPU: Intel i7-10700K or AMD Ryzen 7 3700X
  - GPU: GTX 1660 Super 6GB or RTX 2060 6GB
  - RAM: 16GB DDR4
  - Storage: 512GB NVMe SSD + 1TB HDD
  - Cost: ~$600-800

- **Accessories**:
  - USB webcam (Logitech C920) - $60
  - Basic robot platform (TurtleBot3 Burger kit) - $200-300
  - Total: ~$860-1,360 (staying under $1k with careful selection)

**Capabilities**:
- Full Isaac Sim simulation
- ROS 2 development and testing
- Basic computer vision tasks
- Algorithm development and validation

### Mid-range Tier ($3-5k) - "Hybrid Learning" Approach

**Target Audience**: Students wanting both simulation and limited physical robot experience

**Core Components**:
- **Computer**: High-performance workstation
  - CPU: Intel i9-12900K or AMD Ryzen 9 5900X
  - GPU: RTX 3070 8GB or RTX 3080 10GB
  - RAM: 32GB DDR4-3200
  - Storage: 1TB NVMe SSD
  - Case, PSU, Motherboard: Quality components
  - Cost: ~$1,800-2,200

- **Jetson Platform**:
  - NVIDIA Jetson Orin Nano Developer Kit - $399
  - Power adapter and accessories - $50

- **Robot Platform**:
  - TurtleBot3 Waffle Pi or upgraded Burger - $800-1,200
  - Additional sensors (LIDAR, camera) - $200-300

- **Accessories**:
  - 3D printer (Ender 3 V2) - $200
  - Basic electronics kit - $100
  - Tools and cables - $100

**Total**: ~$3,049-$4,149

**Capabilities**:
- Full simulation capabilities
- Physical robot deployment
- Edge AI inference testing
- Hardware-in-the-loop validation

### Premium Tier ($15k+) - "Full Physical AI Experience"

**Target Audience**: Advanced students, research labs, institutions seeking complete humanoid robotics experience

**Core Components**:
- **Workstation**:
  - CPU: AMD Threadripper PRO 5975WX or Intel Xeon
  - GPU: RTX 4090 24GB or dual RTX 4080
  - RAM: 64GB+ ECC DDR4
  - Storage: 2TB+ NVMe SSD RAID 0
  - Water cooling, premium PSU, etc.
  - Cost: ~$4,000-6,000

- **Jetson Platforms**:
  - 2x NVIDIA Jetson Orin AGX (64GB) - $2,000
  - 2x Jetson Orin Nano - $800
  - Accessories and power supplies - $200

- **Robot Platforms**:
  - Unitree Go1 quadruped (educational discount) - $20,000-25,000
  - OR Unitree G1 humanoid (when available) - $40,000-50,000
  - Alternative: Poppy Ergo Jr (for manipulation) - $1,500
  - Additional manipulators and sensors - $2,000-5,000

- **Simulation Hardware**:
  - VR headset (Meta Quest Pro) for immersive simulation - $800
  - Motion capture system (OptiTrack Flex 3) - $3,000-5,000

- **Infrastructure**:
  - Lab furniture and safety equipment - $2,000
  - Networking equipment - $500
  - Workshop tools and equipment - $1,000

**Total**: $30,000-$80,000+ depending on exact configuration

**Capabilities**:
- Full simulation-to-reality transfer
- Physical humanoid robot operation
- Advanced perception and control
- Research-level experimentation

## Cloud-Native Fallback Documentation

### AWS/NVIDIA Omniverse Solution

For students unable to afford local hardware, we provide a cloud-native fallback:

**AWS Cloud Credits Program**:
- Students can apply for AWS Cloud Credits for Research
- Estimated cost: $100-300/quarter per student
- Access to high-end GPU instances (p4d.24xlarge with A100 GPUs)

**NVIDIA Omniverse Cloud**:
- Isaac Sim on NVIDIA Omniverse Cloud
- Access to high-fidelity simulation
- Integration with AWS for compute

**Performance Requirements**:
- Remote desktop solution (Parsec, Teradici, etc.)
- Stable internet connection (50+ Mbps)
- Local machine requirements: Basic laptop capable of streaming

**Cost Breakdown for Cloud Solution**:
- AWS EC2 p4d.24xlarge: ~$3.00/hour
- For 10 hours/week: ~$1,200/quarter
- With AWS Educate credits: $300-600/quarter
- Additional costs for storage and data transfer

### Ensuring Student Cost Under $300/Quarter

To keep total student cost under $300/quarter:

1. **Hardware Rental Program**: Partner with vendors for student discounts
2. **University Lab Access**: Leverage existing institutional hardware
3. **Group Projects**: Share hardware costs among team members
4. **Cloud Credits**: Secure educational grants and credits
5. **Open Source Alternatives**: Maximize use of free tools and platforms

## 2026 Availability Validation

### Component Availability Research

**Confirmed Available (Q1 2026)**:
- NVIDIA Jetson Orin Nano: In production, widely available
- Isaac Sim 2024.2+: Continuously updated, long-term support
- ROS 2 Jazzy: Released April 2024, 2-year support cycle
- Unitree robots: Available through educational partnerships

**Potentially Unavailable (Risk Factors)**:
- Unitree G1 humanoid: New product, limited availability
- Specific sensor models: May be discontinued

**Mitigation Strategies**:
- Multiple supplier options for each component
- Compatible alternative models identified
- Simulation-first approach for unavailable hardware
- Strong industry partnerships for educational pricing

## Performance Validation Requirements

### Hardware Specifications for Course Requirements

**Minimum Requirements**:
- GPU: NVIDIA RTX 3060 or equivalent (8GB+ VRAM)
- CPU: 6+ cores, 3.0+ GHz per core
- RAM: 16GB minimum
- Storage: 500GB SSD minimum
- OS: Ubuntu 22.04 LTS

**Recommended Requirements**:
- GPU: NVIDIA RTX 4070 or Jetson Orin Nano
- CPU: 8+ cores, 3.5+ GHz per core
- RAM: 32GB
- Storage: 1TB SSD
- OS: Ubuntu 22.04 LTS with ROS 2 Jazzy

### Performance Benchmarks

All configurations must achieve:
- Isaac Sim: 30+ FPS with basic humanoid robot
- Perception pipeline: ≥15 Hz real-time inference
- Control loop: ≥100 Hz for stable control
- End-to-end pipeline: ≤500ms latency

## Vendor Partnerships and Educational Pricing

### Confirmed Educational Partners

**NVIDIA**:
- Isaac Sim educational licenses
- Jetson developer program pricing
- Cloud credits for research

**Unitree**:
- Educational discounts for robots
- Academic partnership program
- Extended warranty options

**ROS-Industrial**:
- Training materials and support
- Certification programs
- Community resources

## Implementation Timeline

### Q1 2026 - Pre-Launch
- Finalize vendor partnerships
- Procure pilot lab equipment
- Validate all configurations
- Develop setup tutorials

### Q2 2026 - Soft Launch
- Pilot course with limited enrollment
- Iterate based on feedback
- Refine hardware recommendations
- Prepare full launch materials

### Q3 2026 - Full Launch
- Course available to full enrollment
- All lab configurations validated
- Cloud fallback operational
- Support infrastructure in place

## Conclusion

This flexible lab setup approach ensures that students can participate in the Physical AI & Humanoid Robotics course regardless of their budget constraints, while still achieving the same learning outcomes. The simulation-first approach allows for comprehensive learning even with limited hardware, while the premium tier offers advanced students the opportunity to work with cutting-edge humanoid platforms.

The cloud-native fallback ensures universal access, and our partnerships with vendors help keep costs manageable for students. All configurations are designed to meet the course's performance requirements while providing a pathway for students to advance from simulation to real hardware as their resources allow.
