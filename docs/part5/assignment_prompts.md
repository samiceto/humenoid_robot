# Graded Assignment Prompts for Physical AI & Humanoid Robotics Course

## Assignment 1: Development Environment Setup and ROS 2 Basics
**Week**: 1
**Type**: Individual
**Weight**: 5%
**Due Date**: End of Week 1

### Objective
Students will set up the development environment and demonstrate basic ROS 2 programming skills.

### Requirements
1. Successfully install and configure Ubuntu 22.04 LTS
2. Install ROS 2 Iron with all required dependencies
3. Create a ROS 2 package with a publisher and subscriber node
4. Implement a custom message type that exchanges robot sensor data
5. Demonstrate the publisher/subscriber communication with a working example
6. Document the setup process with troubleshooting tips

### Deliverables
- ROS 2 package with source code
- Setup documentation (PDF format)
- Video demonstration of working publisher/subscriber
- Installation guide for peers

### Evaluation Criteria
- Functionality: 40% (Code compiles and runs correctly)
- Documentation: 30% (Clear setup instructions and explanations)
- Code Quality: 20% (Well-structured, commented code)
- Presentation: 10% (Video demonstration quality)

---

## Assignment 2: URDF Robot Model Creation
**Week**: 2
**Type**: Individual
**Weight**: 8%
**Due Date**: End of Week 2

### Objective
Students will create a complete URDF model of a simple robot and validate it in simulation.

### Requirements
1. Design a robot with at least 6 degrees of freedom
2. Include proper kinematic chain with joints and links
3. Add visual and collision geometry for all links
4. Include inertial properties for dynamic simulation
5. Add at least 2 sensors (camera, IMU, or LIDAR)
6. Validate the model using check_urdf tool
7. Test the model in RViz for visualization

### Deliverables
- Complete URDF file
- Supporting mesh files (if applicable)
- Validation report
- Screenshots of model in RViz
- Technical documentation

### Evaluation Criteria
- Model Completeness: 35% (All required components included)
- Kinematic Correctness: 25% (Proper joint definitions and connections)
- Validation: 20% (Model passes all checks)
- Documentation: 15% (Clear technical documentation)
- Creativity: 5% (Innovative design elements)

---

## Assignment 3: Isaac Sim Scene Creation and Robot Integration
**Week**: 2
**Type**: Individual
**Weight**: 7%
**Due Date**: End of Week 2

### Objective
Students will create a simulation environment and integrate their robot model.

### Requirements
1. Create a complex scene with multiple objects and obstacles
2. Import the URDF robot model into Isaac Sim
3. Configure physics properties for the robot and environment
4. Set up at least 2 sensors on the robot (camera and IMU)
5. Implement basic movement controls for the robot
6. Document sensor data acquisition and processing
7. Validate that the robot behaves physically correctly

### Deliverables
- Isaac Sim scene file
- Robot configuration files
- Sensor data acquisition code
- Performance validation report
- Video demonstration of robot in scene

### Evaluation Criteria
- Scene Complexity: 30% (Interesting and challenging environment)
- Robot Integration: 25% (Proper physics and sensor configuration)
- Sensor Functionality: 25% (Sensors work correctly and provide data)
- Validation: 15% (Physical behavior verified)
- Documentation: 5% (Clear setup and operation instructions)

---

## Assignment 4: Advanced Simulation with Domain Randomization
**Week**: 3
**Type**: Individual
**Weight**: 10%
**Due Date**: End of Week 3

### Objective
Students will implement domain randomization techniques to improve simulation robustness.

### Requirements
1. Implement domain randomization for at least 3 environmental parameters
2. Create a perception task (object detection or localization)
3. Train a model in the randomized simulation environment
4. Test the model's performance across different domain conditions
5. Compare performance with and without domain randomization
6. Analyze the effectiveness of different randomization strategies
7. Document the implementation and results

### Deliverables
- Domain randomization code
- Trained perception model
- Performance comparison analysis
- Technical report on randomization strategies
- Visualization of results

### Evaluation Criteria
- Randomization Implementation: 30% (Effective parameter randomization)
- Performance Analysis: 25% (Thorough comparison and analysis)
- Technical Quality: 20% (Proper implementation and validation)
- Results Documentation: 15% (Clear presentation of findings)
- Innovation: 10% (Creative approaches to randomization)

---

## Assignment 5: Isaac ROS Perception Pipeline Implementation
**Week**: 4
**Type**: Individual
**Weight**: 12%
**Due Date**: End of Week 4

### Objective
Students will implement a complete perception pipeline using Isaac ROS packages.

### Requirements
1. Integrate at least 3 Isaac ROS perception packages (AprilTag, Visual SLAM, Bi3D, or CenterPose)
2. Create a ROS 2 launch file that starts the complete pipeline
3. Implement data fusion from multiple perception modules
4. Achieve real-time performance (≥15 Hz) for the complete pipeline
5. Validate accuracy of perception outputs
6. Create a visualization of perception results
7. Document performance metrics and optimization strategies

### Deliverables
- Complete perception pipeline code
- ROS 2 launch files
- Performance benchmarking results
- Accuracy validation report
- Visualization tools and outputs

### Evaluation Criteria
- Pipeline Integration: 30% (All components work together)
- Real-time Performance: 25% (Meets ≥15 Hz requirement)
- Accuracy: 20% (Perception results are accurate)
- Documentation: 15% (Clear implementation and results documentation)
- Optimization: 10% (Effective performance optimization)

---

## Assignment 6: Jetson Orin Deployment and Optimization
**Week**: 5
**Type**: Individual
**Weight**: 10%
**Due Date**: End of Week 5

### Objective
Students will deploy and optimize a perception pipeline on Jetson Orin hardware.

### Requirements
1. Deploy the perception pipeline from Assignment 5 on Jetson Orin
2. Optimize the pipeline for real-time performance on embedded hardware
3. Use TensorRT for neural network acceleration
4. Achieve ≥15 Hz inference on Jetson Orin
5. Monitor and report resource utilization (CPU, GPU, memory)
6. Compare performance with simulation results
7. Document optimization techniques and results

### Deliverables
- Optimized perception pipeline for Jetson
- TensorRT optimized models
- Performance benchmarking on Jetson
- Resource utilization reports
- Comparison with simulation results

### Evaluation Criteria
- Deployment Success: 30% (Pipeline runs correctly on Jetson)
- Performance Optimization: 25% (Achieves ≥15 Hz requirement)
- Resource Management: 20% (Efficient resource utilization)
- Comparison Analysis: 15% (Thorough performance comparison)
- Documentation: 10% (Clear optimization documentation)

---

## Assignment 7: LLM Integration for Robot Command Interpretation
**Week**: 6
**Type**: Individual
**Weight**: 10%
**Due Date**: End of Week 6

### Objective
Students will integrate a Large Language Model to interpret natural language commands for robot control.

### Requirements
1. Integrate an LLM API (OpenAI, Claude, or similar) with ROS 2
2. Implement natural language command parsing and interpretation
3. Create a mapping from natural language to robot actions
4. Implement safety checks to validate LLM outputs before execution
5. Test with at least 10 different command types
6. Document the integration architecture and safety measures
7. Evaluate accuracy and safety of the system

### Deliverables
- LLM integration code
- Command interpretation system
- Safety validation framework
- Testing results and evaluation
- Architecture documentation

### Evaluation Criteria
- Integration Quality: 30% (Proper LLM integration with ROS 2)
- Command Interpretation: 25% (Accurate command parsing and mapping)
- Safety Implementation: 20% (Effective safety checks and validation)
- Testing: 15% (Comprehensive testing with various commands)
- Documentation: 10% (Clear system documentation)

---

## Assignment 8: Bipedal Walking Controller Implementation
**Week**: 7
**Type**: Individual
**Weight**: 12%
**Due Date**: End of Week 7

### Objective
Students will implement a stable bipedal walking controller for a humanoid robot.

### Requirements
1. Implement a walking controller using ZMP or inverted pendulum approach
2. Create stable walking gait with adjustable parameters (speed, step length)
3. Implement balance recovery mechanisms for small disturbances
4. Test the controller in simulation with realistic physics
5. Achieve stable walking for at least 100 steps
6. Document the control strategy and parameter tuning process
7. Analyze stability margins and performance metrics

### Deliverables
- Walking controller implementation
- Parameter tuning documentation
- Stability analysis report
- Performance metrics and validation
- Video demonstration of walking

### Evaluation Criteria
- Walking Stability: 35% (Achieves stable walking for required duration)
- Control Strategy: 25% (Effective control algorithm implementation)
- Balance Recovery: 20% (Handles disturbances appropriately)
- Performance Analysis: 15% (Thorough metrics and analysis)
- Documentation: 5% (Clear implementation documentation)

---

## Assignment 9: Whole-Body Controller for Manipulation
**Week**: 8
**Type**: Individual
**Weight**: 12%
**Due Date**: End of Week 8

### Objective
Students will implement a whole-body controller that coordinates manipulation and balance.

### Requirements
1. Implement a whole-body controller using optimization-based approach
2. Coordinate arm manipulation with balance maintenance
3. Handle multi-contact scenarios (hands and feet)
4. Integrate force control for safe interaction
5. Test with at least 3 different manipulation tasks
6. Achieve real-time performance (≥100 Hz control loop)
7. Document the optimization formulation and solution

### Deliverables
- Whole-body controller implementation
- Optimization solver integration
- Task coordination code
- Performance benchmarking
- Testing results and analysis

### Evaluation Criteria
- Controller Implementation: 30% (Proper whole-body control formulation)
- Task Coordination: 25% (Effective manipulation-balance coordination)
- Real-time Performance: 20% (Meets 100 Hz requirement)
- Multi-contact Handling: 15% (Proper contact force management)
- Testing: 10% (Comprehensive task testing)

---

## Assignment 10: Adaptive Control System
**Week**: 9
**Type**: Individual
**Weight**: 10%
**Due Date**: End of Week 9

### Objective
Students will implement an adaptive control system that learns and adjusts to changing conditions.

### Requirements
1. Implement an adaptive control algorithm (MRAC or similar)
2. Demonstrate adaptation to parameter changes or disturbances
3. Ensure stability during adaptation process
4. Test with at least 2 different adaptation scenarios
5. Compare performance with non-adaptive controller
6. Document adaptation mechanism and stability guarantees
7. Analyze convergence properties

### Deliverables
- Adaptive controller implementation
- Adaptation mechanism code
- Stability analysis
- Performance comparison results
- Convergence analysis

### Evaluation Criteria
- Adaptation Quality: 35% (Effective parameter adaptation)
- Stability: 25% (Maintains stability during adaptation)
- Performance Improvement: 20% (Better performance than fixed controller)
- Analysis: 15% (Thorough stability and convergence analysis)
- Documentation: 5% (Clear implementation documentation)

---

## Assignment 11: System Integration Project
**Week**: 10
**Type**: Individual
**Weight**: 15%
**Due Date**: End of Week 10

### Objective
Students will integrate perception, planning, control, and learning components into a unified system.

### Requirements
1. Integrate at least 4 major components from previous assignments
2. Create a modular software architecture with clear interfaces
3. Implement error handling and recovery mechanisms
4. Achieve real-time performance requirements
5. Document the system architecture and integration process
6. Validate the integrated system with comprehensive testing
7. Demonstrate the system performing a complex task

### Deliverables
- Integrated system codebase
- Architecture documentation
- Testing framework and results
- Performance validation report
- Demonstration video

### Evaluation Criteria
- Integration Quality: 30% (All components work together seamlessly)
- Architecture Design: 25% (Well-designed modular architecture)
- Performance: 20% (Meets real-time requirements)
- Testing: 15% (Comprehensive validation and testing)
- Documentation: 10% (Clear system documentation)

---

## Assignment 12: Capstone Project Planning
**Week**: 11
**Type**: Individual
**Weight**: 8%
**Due Date**: End of Week 11 (Planning Phase)

### Objective
Students will plan and design their comprehensive capstone project.

### Requirements
1. Define a complex humanoid robotics task or application
2. Create a detailed project plan with timeline and milestones
3. Identify required components and technologies from course
4. Design the system architecture for the capstone project
5. Plan testing and validation procedures
6. Identify potential risks and mitigation strategies
7. Create a preliminary implementation approach

### Deliverables
- Project proposal document
- Detailed project plan with timeline
- System architecture design
- Risk analysis and mitigation plan
- Implementation approach document

### Evaluation Criteria
- Project Scope: 30% (Appropriate complexity and scope)
- Planning Quality: 25% (Detailed and realistic plan)
- Architecture Design: 20% (Well-designed system architecture)
- Risk Management: 15% (Identified risks and mitigation)
- Feasibility: 10% (Realistic timeline and approach)

---

## Assignment 13: Capstone Project Implementation and Presentation
**Week**: 11-12
**Type**: Individual
**Weight**: 20%
**Due Date**: End of Week 12

### Objective
Students will implement, test, and present their comprehensive capstone project.

### Requirements
1. Implement the complete capstone project as planned
2. Integrate multiple course concepts and technologies
3. Achieve the defined project objectives
4. Conduct thorough testing and validation
5. Prepare and deliver a technical presentation
6. Document the complete implementation
7. Demonstrate the working system

### Deliverables
- Complete capstone project implementation
- Technical documentation
- Testing and validation results
- Technical presentation (slides and video)
- Source code and assets

### Evaluation Criteria
- Implementation Quality: 30% (Complete and working implementation)
- Integration: 25% (Effective integration of multiple concepts)
- Innovation: 20% (Creative and novel approaches)
- Presentation: 15% (Clear and engaging technical presentation)
- Documentation: 10% (Comprehensive project documentation)

---

## Assignment 14: Deployment and Operational Validation
**Week**: 12
**Type**: Individual
**Weight**: 10%
**Due Date**: End of Week 12

### Objective
Students will deploy their system on physical hardware and validate operational performance.

### Requirements
1. Deploy the system on physical robot hardware
2. Validate performance compared to simulation
3. Document deployment challenges and solutions
4. Test operational procedures and safety measures
5. Monitor system performance during extended operation
6. Analyze reality gap and adaptation requirements
7. Create operational manual for the system

### Deliverables
- Physical deployment implementation
- Performance comparison report
- Deployment documentation
- Operational validation results
- Operational manual

### Evaluation Criteria
- Deployment Success: 30% (Successful physical deployment)
- Performance Validation: 25% (Thorough comparison and analysis)
- Operational Procedures: 20% (Effective operational protocols)
- Reality Gap Analysis: 15% (Analysis of sim-to-real differences)
- Documentation: 10% (Clear operational documentation)

---

## General Assignment Guidelines

### Submission Requirements
- All code must be properly documented with comments
- Documentation should be in PDF format
- Videos should be in MP4 format (maximum 5 minutes)
- Code must be submitted in a structured format with clear README
- All deliverables must be submitted through the course management system

### Late Submission Policy
- Assignments submitted 1-2 days late: 10% penalty
- Assignments submitted 3-7 days late: 25% penalty
- Assignments submitted more than 7 days late: Not accepted without prior approval

### Academic Integrity
- All work must be original and properly cited
- Collaboration is allowed but must be documented
- Code from external sources must be attributed
- Any form of plagiarism will result in a zero grade

### Technical Requirements
- All code must run on the specified development environment
- Performance requirements must be met as specified
- Documentation must be clear and comprehensive
- Testing must be thorough and well-documented