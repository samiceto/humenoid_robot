---
sidebar_position: 3
title: "Graded Assignments: Weeks 1-3"
description: "Three graded assignment prompts with detailed rubrics for the first 3 weeks"
---

# Graded Assignments: Weeks 1-3

## Assignment 1: Physical AI Concepts and Course Setup
**Due Date**: End of Week 1
**Weight**: 15% of final grade
**Estimated Time**: 8-10 hours

### Assignment Description
This assignment introduces you to the core concepts of Physical AI and requires you to set up your development environment for the course.

### Tasks
1. **Concept Analysis** (50 points)
   - Write a 500-word essay explaining the difference between traditional AI and Physical AI
   - Identify three real-world applications where Physical AI is essential
   - Discuss the sim-to-real challenge and its importance in robotics

2. **Environment Setup** (30 points)
   - Install Ubuntu 22.04 LTS (or verify existing installation)
   - Install ROS 2 Iron with all required dependencies
   - Install Isaac Sim 2024.2+ and verify basic functionality
   - Document your installation process with screenshots

3. **Technology Stack Familiarization** (20 points)
   - Research one component from the course technology stack (Isaac ROS, Nav2, OpenVLA, etc.)
   - Write a 200-word summary of its purpose and application in humanoid robotics

### Rubric

| Criteria | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (60-69%) | Unsatisfactory (0-59%) |
|----------|-------------------|---------------|---------------------|--------------------------|----------------------|
| Concept Analysis | Comprehensive understanding with clear explanations and examples | Good understanding with minor gaps | Basic understanding with some explanation | Limited understanding with significant gaps | Poor understanding or incorrect information |
| Environment Setup | All components installed and verified with clear documentation | Most components installed with good documentation | Basic setup completed with adequate documentation | Some components installed with limited documentation | Incomplete setup or poor documentation |
| Technology Research | Thorough research with detailed explanation and applications | Good research with clear explanation | Adequate research with basic explanation | Limited research with minimal explanation | Insufficient research or unclear explanation |
| Writing Quality | Clear, concise, well-organized with proper citations | Generally clear with minor issues | Adequately clear with some organization issues | Unclear writing with organization problems | Poor writing quality, difficult to understand |

### Submission Requirements
- Submit a single PDF document containing all components
- Include screenshots of your environment setup
- Cite all sources used in your research
- Follow the course formatting guidelines

---

## Assignment 2: ROS 2 Fundamentals Implementation
**Due Date**: End of Week 2
**Weight**: 25% of final grade
**Estimated Time**: 12-15 hours

### Assignment Description
This assignment requires you to implement basic ROS 2 concepts by creating custom nodes and establishing communication between them.

### Tasks
1. **Custom Node Development** (60 points)
   - Create a ROS 2 package called `robotics_fundamentals`
   - Implement a publisher node that publishes sensor data (simulated IMU, joint positions)
   - Implement a subscriber node that processes the sensor data and publishes a simple analysis
   - Create a service server that provides robot status information
   - Create a service client that requests and displays robot status

2. **Message Design** (25 points)
   - Define custom message types for your sensor data
   - Define custom service types for robot status queries
   - Ensure proper message structure and documentation

3. **Launch File Configuration** (15 points)
   - Create a launch file that starts all your nodes simultaneously
   - Configure parameters appropriately
   - Include comments explaining the launch configuration

### Rubric

| Criteria | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (60-69%) | Unsatisfactory (0-59%) |
|----------|-------------------|---------------|---------------------|--------------------------|----------------------|
| Node Implementation | All nodes function correctly with advanced features | All nodes function correctly with minor issues | All nodes function with basic features | Most nodes function with some issues | Nodes do not function properly |
| Message Design | Well-designed custom messages with proper structure and documentation | Good message design with minor issues | Adequate message design with basic documentation | Basic message design with minimal documentation | Poor message design or incorrect structure |
| Launch Configuration | Launch file works perfectly with proper parameter configuration | Launch file works with minor configuration issues | Launch file works with basic configuration | Launch file has issues but mostly works | Launch file does not work properly |
| Code Quality | Clean, well-documented, follows ROS 2 best practices | Good code quality with minor issues | Adequate code quality with basic documentation | Code has issues but is functional | Poor code quality, difficult to follow |
| Functionality | All requirements met with additional advanced features | All requirements met with minor issues | All requirements met with basic functionality | Most requirements met with some issues | Requirements not met |

### Submission Requirements
- Submit a ZIP file containing your ROS 2 package
- Include a README.md file with instructions for building and running your code
- Provide a brief video demonstration (3-5 minutes) showing your nodes in action
- Include screenshots of successful node communication

---

## Assignment 3: URDF Robot Modeling
**Due Date**: End of Week 3
**Weight**: 30% of final grade
**Estimated Time**: 15-20 hours

### Assignment Description
This assignment requires you to design and model a simple humanoid robot using URDF, including proper kinematic structure and validation.

### Tasks
1. **Robot Design** (50 points)
   - Create a URDF file for a simple humanoid robot with at least 12 degrees of freedom
   - Include torso, head, two arms, and two legs with appropriate joints
   - Define proper visual, collision, and inertial properties for each link
   - Ensure realistic joint limits and ranges of motion

2. **Kinematic Validation** (25 points)
   - Validate your URDF using `check_urdf` command
   - Test kinematic structure using appropriate ROS 2 tools
   - Demonstrate forward kinematics by setting joint positions and observing end-effector positions

3. **Xacro Implementation** (15 points)
   - Convert your URDF to Xacro format
   - Use macros to reduce code duplication (especially for symmetrical parts like arms and legs)
   - Include parameters for easy customization

4. **Simulation Integration** (10 points)
   - Verify that your robot model works in a simulation environment (Gazebo or Isaac Sim)
   - Provide screenshots or video of the model in simulation

### Rubric

| Criteria | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (60-69%) | Unsatisfactory (0-59%) |
|----------|-------------------|---------------|---------------------|--------------------------|----------------------|
| Robot Design | Sophisticated design with realistic kinematics and properties | Good design with realistic elements | Adequate design meeting basic requirements | Basic design with some unrealistic elements | Poor design not meeting requirements |
| Kinematic Structure | Proper kinematic chain with realistic DOF and joint types | Good kinematic structure with minor issues | Adequate kinematic structure meeting requirements | Basic kinematic structure with issues | Poor kinematic structure |
| URDF Validation | Fully validated with no errors and proper testing | Mostly validated with minor warnings | Adequately validated meeting requirements | Basic validation with some errors | Poor validation or not validated |
| Xacro Implementation | Effective use of macros and parameters with no duplication | Good use of Xacro features with minor duplication | Adequate use of Xacro meeting requirements | Basic use of Xacro with significant duplication | Poor or no Xacro implementation |
| Simulation Integration | Works perfectly in simulation with full functionality | Works well in simulation with minor issues | Works in simulation meeting requirements | Basic functionality in simulation | Does not work in simulation |
| Documentation | Comprehensive with clear explanations and diagrams | Good documentation with minor gaps | Adequate documentation meeting requirements | Basic documentation with gaps | Poor or no documentation |

### Submission Requirements
- Submit a ZIP file containing your URDF/Xacro files and any mesh files
- Include a README.md file with instructions for viewing and validating your robot model
- Provide screenshots of the `check_urdf` output
- Include a video demonstration of your robot in simulation
- Document any design decisions and assumptions made during the modeling process

---

## General Assignment Guidelines

### Late Submission Policy
- Late assignments will be penalized 5% per day (including weekends)
- No submissions accepted more than 1 week late without prior approval
- Technical issues are not an acceptable reason for late submission

### Collaboration Policy
- Individual assignments must be completed independently
- You may discuss concepts with classmates but not share code
- Properly cite any external resources used
- Code that appears to be copied will result in a zero for the assignment

### Grading Scale
- A: 90-100%
- B: 80-89%
- C: 70-79%
- D: 60-69%
- F: Below 60%

### Submission Format
- All written components should be in PDF format
- Code should be properly formatted and commented
- Videos should be in MP4 format (maximum 100MB)
- ZIP files should not exceed 500MB
- Follow the naming convention: LastName_FirstName_AssignmentX.zip

### Technical Support
- Use the course discussion forum for assignment-related questions
- Office hours are available for more complex technical issues
- TA support is available for environment and tool-related problems