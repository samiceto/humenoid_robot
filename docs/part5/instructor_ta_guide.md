# Instructor and TA Guide for Physical AI & Humanoid Robotics Course

## Overview

This guide provides comprehensive information for instructors and teaching assistants to effectively deliver the Physical AI & Humanoid Robotics course. The course is designed to provide students with a complete foundation in humanoid robotics, progressing from basic concepts to advanced implementation over 13 weeks.

## Course Structure and Timeline

### Course Duration
- **Total Weeks**: 13 weeks
- **Class Sessions**: 2-3 sessions per week (6-9 hours/week)
- **Total Contact Hours**: ~100 hours
- **Course Level**: Advanced undergraduate/graduate level

### Weekly Breakdown
| Week | Topic | Key Components | Student Deliverables |
|------|-------|----------------|---------------------|
| 1 | Introduction & ROS 2 Basics | Environment setup, basic programming | Setup documentation, basic ROS nodes |
| 2 | Robot Modeling & Simulation | URDF, Isaac Sim | Robot model, simulation scene |
| 3 | Advanced Simulation | Domain randomization | Randomization implementation |
| 4 | Isaac ROS Perception | AprilTag, Visual SLAM, Bi3D | Perception pipeline |
| 5 | Edge Computing & Cognitive Architectures | Jetson Orin, cognitive systems | Optimized pipeline, architecture |
| 6 | LLM Integration & Vision-Language | Natural language, multimodal systems | LLM integration, vision-language |
| 7 | Bipedal Locomotion | Walking controllers | Walking controller |
| 8 | Whole-Body Control | Multi-task optimization | Whole-body controller |
| 9 | Adaptive & Learning-Based Control | Machine learning, adaptation | Learning-based controller |
| 10 | System Integration | Component integration | Integrated system |
| 11 | Capstone Project | Comprehensive implementation | Project planning |
| 12 | Deployment & Operation | Physical deployment | Deployed system |
| 13 | Review & Assessment | Final evaluation | Final presentation |

## Instructor Preparation Guidelines

### Before Each Week

#### Week 1: Introduction & ROS 2 Basics
**Preparation Time**: 4-6 hours
- Review Ubuntu 22.04 installation procedures
- Test ROS 2 Iron installation on multiple machines
- Prepare troubleshooting guides for common setup issues
- Set up virtual machines for students without proper hardware
- Review basic ROS 2 concepts and examples

**Materials to Prepare**:
- Installation guides and troubleshooting documents
- Sample ROS 2 publisher/subscriber code
- Virtual machine images for students with limited hardware
- List of recommended hardware configurations

#### Week 2: Robot Modeling & Simulation
**Preparation Time**: 5-7 hours
- Install and test Isaac Sim on instructor machine
- Prepare sample URDF models for demonstration
- Set up Isaac Sim scenes for class examples
- Test URDF validation tools and processes
- Prepare simulation performance optimization strategies

**Materials to Prepare**:
- Sample URDF files and 3D models
- Isaac Sim scene files
- URDF validation scripts
- Performance optimization guides

#### Week 3: Advanced Simulation
**Preparation Time**: 4-6 hours
- Implement domain randomization examples
- Prepare performance comparison tools
- Set up simulation-to-reality transfer demonstrations
- Test different randomization strategies
- Prepare analysis tools for student evaluations

**Materials to Prepare**:
- Domain randomization code examples
- Performance comparison tools
- Simulation-to-reality validation frameworks
- Analysis scripts for evaluation

#### Week 4: Isaac ROS Perception
**Preparation Time**: 6-8 hours
- Install and test all Isaac ROS packages
- Prepare perception pipeline examples
- Set up performance monitoring tools
- Test ≥15 Hz performance requirements
- Prepare debugging tools for perception issues

**Materials to Prepare**:
- Isaac ROS installation scripts
- Perception pipeline examples
- Performance monitoring tools
- Debugging and troubleshooting guides

#### Week 5: Edge Computing & Cognitive Architectures
**Preparation Time**: 5-7 hours
- Test Jetson Orin deployment procedures
- Prepare TensorRT optimization examples
- Set up cognitive architecture implementations
- Test ≥15 Hz performance on embedded hardware
- Prepare resource monitoring tools

**Materials to Prepare**:
- Jetson deployment scripts
- TensorRT optimization examples
- Cognitive architecture templates
- Performance monitoring tools

#### Week 6: LLM Integration & Vision-Language
**Preparation Time**: 5-7 hours
- Set up LLM API access for demonstrations
- Prepare safety and validation frameworks
- Test vision-language integration examples
- Prepare ethical considerations materials
- Set up multimodal integration examples

**Materials to Prepare**:
- LLM integration code examples
- Safety validation frameworks
- Vision-language system examples
- Ethical guidelines and considerations

#### Week 7: Bipedal Locomotion
**Preparation Time**: 6-8 hours
- Prepare ZMP and inverted pendulum examples
- Set up walking controller demonstrations
- Prepare balance recovery examples
- Test stability analysis tools
- Prepare gait analysis frameworks

**Materials to Prepare**:
- Walking controller examples
- Balance recovery demonstrations
- Stability analysis tools
- Gait analysis frameworks

#### Week 8: Whole-Body Control
**Preparation Time**: 6-8 hours
- Prepare optimization-based control examples
- Set up multi-task coordination demonstrations
- Test real-time performance (≥100 Hz) requirements
- Prepare multi-contact scenario examples
- Prepare force control integration examples

**Materials to Prepare**:
- Whole-body control frameworks
- Multi-task coordination examples
- Real-time performance tools
- Multi-contact scenario demonstrations

#### Week 9: Adaptive & Learning-Based Control
**Preparation Time**: 6-8 hours
- Prepare adaptive control examples
- Set up reinforcement learning demonstrations
- Test safety in learning systems
- Prepare imitation learning examples
- Prepare transfer learning demonstrations

**Materials to Prepare**:
- Adaptive control implementations
- Reinforcement learning examples
- Safety validation tools
- Imitation learning frameworks

#### Week 10: System Integration
**Preparation Time**: 5-7 hours
- Prepare integration architecture examples
- Set up modular design demonstrations
- Test real-time system requirements
- Prepare debugging and testing frameworks
- Prepare documentation templates

**Materials to Prepare**:
- System integration frameworks
- Modular architecture examples
- Testing and validation tools
- Documentation templates

#### Week 11: Capstone Project
**Preparation Time**: 4-6 hours
- Prepare project planning templates
- Set up project evaluation criteria
- Prepare milestone tracking tools
- Prepare peer review frameworks
- Prepare presentation guidelines

**Materials to Prepare**:
- Project planning templates
- Evaluation rubrics
- Milestone tracking tools
- Peer review forms

#### Week 12: Deployment & Operation
**Preparation Time**: 5-7 hours
- Prepare physical deployment procedures
- Set up operational validation tools
- Prepare troubleshooting guides for hardware
- Test reality gap analysis tools
- Prepare operational documentation templates

**Materials to Prepare**:
- Deployment procedures
- Validation tools
- Troubleshooting guides
- Operational documentation templates

#### Week 13: Review & Assessment
**Preparation Time**: 4-6 hours
- Prepare comprehensive review materials
- Set up final assessment procedures
- Prepare portfolio evaluation tools
- Prepare career guidance resources
- Prepare feedback collection tools

**Materials to Prepare**:
- Review materials
- Assessment rubrics
- Portfolio evaluation tools
- Career guidance resources

## TA Responsibilities and Guidelines

### General TA Responsibilities
- Assist with lab sessions and practical exercises
- Provide one-on-one support to students
- Grade assignments and provide detailed feedback
- Maintain office hours for student support
- Assist with debugging and troubleshooting
- Monitor student progress and engagement

### Week-Specific TA Guidelines

#### TA Duties for Week 1
- **Lab Support**: Help students with Ubuntu/ROS 2 installation
- **Troubleshooting**: Address common setup issues
- **Code Review**: Review student ROS 2 implementations
- **Documentation**: Assist with setup documentation

**Common Issues to Address**:
- Ubuntu installation problems
- ROS 2 environment setup
- Network and package installation issues
- Basic ROS 2 command understanding

#### TA Duties for Week 2
- **Model Review**: Review student URDF models
- **Simulation Support**: Help with Isaac Sim integration
- **Validation**: Assist with model validation
- **Performance**: Address simulation performance issues

**Common Issues to Address**:
- URDF syntax errors
- Kinematic chain problems
- Visual/collision geometry issues
- Isaac Sim scene setup

#### TA Duties for Week 3
- **Randomization Review**: Evaluate student randomization implementations
- **Performance Analysis**: Help with performance comparisons
- **Optimization**: Assist with simulation optimization
- **Validation**: Support reality gap analysis

**Common Issues to Address**:
- Randomization parameter selection
- Performance degradation
- Validation methodology
- Analysis interpretation

#### TA Duties for Week 4
- **Pipeline Review**: Evaluate perception pipeline implementations
- **Performance Testing**: Assist with ≥15 Hz validation
- **Integration Support**: Help with multi-component integration
- **Debugging**: Address perception system issues

**Common Issues to Address**:
- Component integration problems
- Performance optimization
- Real-time requirements
- Data flow issues

#### TA Duties for Week 5
- **Deployment Support**: Help with Jetson Orin deployment
- **Optimization Review**: Evaluate performance optimizations
- **Architecture Review**: Assess cognitive architecture implementations
- **Performance Validation**: Assist with embedded performance testing

**Common Issues to Address**:
- Jetson deployment issues
- Performance optimization strategies
- Resource management
- Real-time constraints

#### TA Duties for Week 6
- **Integration Support**: Help with LLM integration
- **Safety Review**: Evaluate safety implementation
- **Multimodal Support**: Assist with vision-language integration
- **Ethics Guidance**: Provide guidance on ethical considerations

**Common Issues to Address**:
- LLM API integration
- Safety validation frameworks
- Multimodal data fusion
- Ethical implementation

#### TA Duties for Week 7
- **Controller Review**: Evaluate walking controller implementations
- **Stability Analysis**: Assist with stability validation
- **Balance Support**: Help with balance recovery implementations
- **Performance Testing**: Support gait performance analysis

**Common Issues to Address**:
- Walking stability issues
- Balance control problems
- Gait parameter tuning
- Performance optimization

#### TA Duties for Week 8
- **Integration Support**: Help with whole-body controller integration
- **Optimization Review**: Evaluate multi-task coordination
- **Real-time Testing**: Assist with performance validation
- **Multi-contact Support**: Help with contact handling

**Common Issues to Address**:
- Optimization formulation
- Task prioritization
- Real-time performance
- Contact force management

#### TA Duties for Week 9
- **Adaptation Review**: Evaluate adaptive control implementations
- **Learning Support**: Help with reinforcement learning
- **Safety Validation**: Assist with safe learning systems
- **Convergence Analysis**: Support stability analysis

**Common Issues to Address**:
- Adaptation algorithm implementation
- Learning convergence
- Safety constraint enforcement
- Stability guarantees

#### TA Duties for Week 10
- **Integration Support**: Help with system integration
- **Architecture Review**: Evaluate modular design
- **Performance Testing**: Assist with real-time validation
- **Debugging Support**: Help with complex integration issues

**Common Issues to Address**:
- Component communication
- Real-time constraints
- Modular design principles
- Integration testing

#### TA Duties for Week 11
- **Project Planning**: Assist with capstone project planning
- **Milestone Tracking**: Help with project timeline management
- **Design Review**: Evaluate system architecture
- **Risk Assessment**: Assist with project risk analysis

**Common Issues to Address**:
- Project scope management
- Timeline planning
- Architecture design
- Risk mitigation

#### TA Duties for Week 12
- **Deployment Support**: Help with physical hardware deployment
- **Reality Gap Analysis**: Assist with sim-to-real validation
- **Operational Testing**: Support operational validation
- **Troubleshooting**: Address hardware-specific issues

**Common Issues to Address**:
- Hardware integration
- Reality gap issues
- Operational procedures
- Physical safety considerations

#### TA Duties for Week 13
- **Final Review**: Assist with comprehensive project review
- **Assessment Support**: Help with final evaluation
- **Presentation Support**: Provide presentation feedback
- **Portfolio Review**: Assist with portfolio development

**Common Issues to Address**:
- Comprehensive integration review
- Final project evaluation
- Presentation preparation
- Career planning

## Assessment and Grading Guidelines

### General Grading Philosophy
- Focus on learning outcomes rather than just correctness
- Provide constructive feedback that helps students improve
- Maintain consistency across all evaluators
- Consider effort and improvement in addition to final results

### Assignment Grading Guidelines

#### Programming Assignments (Technical Implementation)
- **Functionality (40-50%)**: Does the code work as intended?
- **Code Quality (20-25%)**: Is the code well-structured and documented?
- **Technical Understanding (20-25%)**: Does the implementation show understanding of concepts?
- **Documentation (10-15%)**: Is the work properly documented?

#### Design Assignments (System Architecture)
- **Design Quality (35-40%)**: Is the design appropriate and well-thought-out?
- **Technical Correctness (30-35%)**: Does the design follow proper principles?
- **Innovation (15-20%)**: Does the design show creative thinking?
- **Documentation (10-15%)**: Is the design properly documented?

#### Analysis Assignments (Performance Evaluation)
- **Analysis Depth (40-45%)**: Is the analysis thorough and insightful?
- **Methodology (25-30%)**: Are appropriate methods used for analysis?
- **Results Interpretation (20-25%)**: Are results properly interpreted?
- **Presentation (10-15%)**: Are results clearly presented?

### Rubric Consistency
- Use the provided detailed rubrics consistently
- Discuss borderline cases with other instructors/TAs
- Calibrate grading with sample assignments
- Provide specific feedback tied to rubric criteria

### Feedback Quality Guidelines
- Be specific rather than general
- Provide actionable suggestions for improvement
- Balance positive feedback with areas for improvement
- Connect feedback to learning objectives
- Use student names and specific examples

## Classroom Management and Engagement

### Encouraging Participation
- Ask open-ended questions about concepts
- Use think-pair-share activities
- Encourage students to explain their implementations
- Create safe spaces for asking questions
- Use real-world examples to illustrate concepts

### Managing Different Skill Levels
- Provide additional resources for students who need foundational support
- Offer advanced challenges for students who finish early
- Use peer mentoring and group work strategically
- Differentiate instruction without tracking students

### Handling Technical Difficulties
- Have backup plans for common technical issues
- Prepare virtual environments for students with limited hardware
- Maintain a repository of common troubleshooting solutions
- Establish clear procedures for reporting technical problems

## Student Support and Resources

### Office Hours Structure
- **Individual Support**: 1:1 help with specific problems
- **Group Review**: Common questions and concepts
- **Code Review**: Feedback on implementations
- **Project Consultation**: Guidance on project work

### Additional Resources
- Online documentation and tutorials
- Video walkthroughs for complex procedures
- Peer mentoring programs
- Study groups and collaboration opportunities

### Accommodation Guidelines
- Work with disability services for appropriate accommodations
- Provide alternative formats when needed
- Offer flexible deadlines when appropriate
- Ensure accessibility in all materials

## Technology and Infrastructure Requirements

### Required Software
- Ubuntu 22.04 LTS (or VM)
- ROS 2 Iron
- Isaac Sim
- Isaac ROS packages
- Development tools (VS Code, Git, etc.)
- Python and C++ development environments

### Hardware Requirements
- Minimum: i7 processor, 16GB RAM, GTX 1060
- Recommended: i9 processor, 32GB RAM, RTX 3070
- GPU with CUDA support for Isaac Sim
- Reliable internet connection

### Backup and Recovery
- Regular backups of student work
- Version control systems (Git)
- Cloud storage for critical materials
- Recovery procedures for system failures

## Communication Protocols

### Instructor-Student Communication
- Email for formal communication
- Discussion boards for technical questions
- Office hours for individual support
- Announcements for important updates

### TA-Student Communication
- Lab sessions for hands-on support
- Office hours for individual help
- Discussion sections for group questions
- Grading feedback for assignment improvement

### Emergency Procedures
- Technical issue escalation procedures
- System failure response protocols
- Alternative delivery methods
- Student notification procedures

## Quality Assurance and Continuous Improvement

### Regular Assessment
- Weekly feedback collection from students
- Peer review of teaching materials
- Self-assessment of teaching effectiveness
- Colleague observation and feedback

### Course Improvement Process
- Collect and analyze student feedback
- Review assignment effectiveness
- Update materials based on technology changes
- Incorporate lessons learned from previous offerings

### Professional Development
- Stay current with robotics research
- Attend relevant conferences and workshops
- Participate in teaching development programs
- Collaborate with industry professionals

## Conclusion

This guide provides the framework for successful delivery of the Physical AI & Humanoid Robotics course. The key to success lies in thorough preparation, consistent application of grading standards, active student engagement, and continuous improvement based on feedback and experience. The course is designed to challenge students while providing them with the support they need to succeed in mastering advanced robotics concepts and implementation skills.