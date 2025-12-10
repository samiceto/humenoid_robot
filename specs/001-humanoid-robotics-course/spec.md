# Feature Specification: Physical AI & Humanoid Robotics Capstone Course

**Feature Branch**: `001-humanoid-robotics-course`
**Created**: 2025-12-09
**Status**: Draft
**Input**: User description: "Complete Specification for the "Physical AI & Humanoid Robotics" Capstone Course phase 1 only, dont touch phase 2 in constitution for now.

Target audience:
- University department heads, lab directors, and bootcamp founders who want to launch a cutting-edge, industry-aligned Physical AI / Humanoid Robotics course in 2026
- Instructors and TAs who will actually teach and support the course

Focus:
- Deliver a fully executable 13-week capstone course that takes students from zero robotics experience to deploying a voice-controlled autonomous humanoid (simulated + optional real hardware)
- Heavy emphasis on the modern 2025–2026 embodied-AI stack: ROS 2 Humble/Iron, NVIDIA Isaac Sim + Isaac ROS, Nav2, Vision-Language-Action models, Jetson Orin edge deployment

Success criteria:
- The final specification package is immediately usable: any competent instructor can run the course in Q1 2026 with no missing pieces
- Contains complete week-by-week syllabus, slide deck outlines, 13+ graded assignment prompts, full grading rubrics, and setup tutorials
- Capstone demo works end-to-end: spoken command → Whisper → LLM planner → ROS 2 action sequence → navigation + manipulation (tested in Isaac Sim and on at least one physical platform)
- Provides three fully costed lab tiers (Budget <$1k, Mid $3–5k, Premium $15k+) with exact part numbers and purchase links that are still valid in 2025
- Includes cloud-native fallback path (AWS/NVIDIA Omniverse) that keeps total student cost under $300/quarter if no local GPU
- Every tool recommended is still officially supported in 2026 and has active community/package maintenance

Constraints:
- Total course length: maximum 13 weeks
- Weekly student time budget: 12–15 hours (including lectures, labs, and project work)
- Prerequisite knowledge: Python, basic deep learning; no prior ROS or robotics required
- Primary OS: Ubuntu 22.04 LTS (all setup instructions must assume Linux)
- GPU requirement for simulation: RTX 4070 Ti / 4080 / 4090 class or equivalent cloud instance
- All assignments must be doable on student-owned hardware or provided lab kits
- Capstone must run at ≥15 Hz real-time inference on Jetson Orin Nano/Orin NX 8–16 GB

Not building:
- A beginner-level robotics or ROS introduction course
- A purely theoretical AI embodiment course with no hardware deployment
- A course focused on non-humanoid platforms (drones, self-driving cars, fixed-base arms as primary target)
- Vendor-specific product marketing (no mandatory use of a particular commercial humanoid)
- Full research survey of all humanoid robots ever built
- Custom LLM training from scratch (use of existing open-source VLA models is allowed)

Deliver the complete, ready-to-launch course specification package in Markdown with embedded file structure."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Launch Complete 13-Week Capstone Course (Priority: P1)

University department heads, lab directors, and bootcamp founders can launch a complete, ready-to-run Physical AI & Humanoid Robotics capstone course that takes students from zero robotics experience to deploying voice-controlled autonomous humanoid systems. The course includes all necessary materials: syllabus, slide decks, assignments, rubrics, and setup tutorials.

**Why this priority**: This is the foundational value proposition - without a complete, executable course package, the entire feature fails to deliver on its primary promise to educational institutions.

**Independent Test**: Course can be fully deployed by an instructor in Q1 2026 with no missing pieces, delivering immediate value to students who progress from zero robotics experience to deploying autonomous humanoid systems.

**Acceptance Scenarios**:

1. **Given** instructor has access to the complete course package, **When** they follow the setup instructions and begin teaching, **Then** students can successfully complete all 13 weeks of content and execute the end-to-end capstone demo.

2. **Given** student has Python and basic deep learning knowledge, **When** they engage with the 13-week curriculum, **Then** they can deploy a voice-controlled autonomous humanoid in simulation and optionally on real hardware.

---
### User Story 2 - Execute End-to-End Voice-Controlled Autonomous System (Priority: P1)

Students can execute a complete end-to-end autonomous humanoid system that processes spoken commands through Whisper → LLM planner → ROS 2 action sequence → navigation + manipulation, working in both Isaac Sim and on at least one physical platform.

**Why this priority**: This represents the core capstone achievement that demonstrates the course's success and the students' mastery of embodied AI concepts.

**Independent Test**: Students can demonstrate the complete pipeline from spoken command to physical robot action, validating the entire embodied AI learning journey.

**Acceptance Scenarios**:

1. **Given** student has completed the course prerequisites and setup, **When** they issue a spoken command to the humanoid system, **Then** the system processes the command through Whisper → LLM planner → ROS 2 → navigation/manipulation and executes the requested action.

2. **Given** the system is running in Isaac Sim, **When** student validates the pipeline, **Then** it performs consistently with real hardware deployment.

---
### User Story 3 - Access Flexible Lab Setup Options (Priority: P2)

Educational institutions and students can choose from three costed lab tiers (Budget <$1k, Mid $3–5k, Premium $15k+) with exact part numbers and purchase links, or use cloud-native fallback that keeps total student cost under $300/quarter.

**Why this priority**: This ensures the course is accessible to institutions and students with varying budget constraints, maximizing adoption potential.

**Independent Test**: Students can successfully implement the course with any of the provided lab configurations or cloud fallback, achieving the same learning outcomes.

**Acceptance Scenarios**:

1. **Given** institution has chosen a lab tier, **When** they purchase the specified equipment using provided part numbers and links, **Then** they can successfully complete all course requirements.

2. **Given** student has limited budget, **When** they use the cloud-native fallback path, **Then** they can still complete all course requirements with total cost under $300/quarter.

---
### User Story 4 - Follow Modern Embodied AI Stack Curriculum (Priority: P2)

Students learn the modern 2025–2026 embodied-AI stack including ROS 2 Humble/Iron, NVIDIA Isaac Sim + Isaac ROS, Nav2, Vision-Language-Action models, and Jetson Orin edge deployment with real-time performance requirements.

**Why this priority**: This ensures students learn industry-relevant, current technologies that will be valuable in the job market in 2026.

**Independent Test**: Students can demonstrate proficiency with the specified technology stack components and meet the performance requirements on target hardware.

**Acceptance Scenarios**:

1. **Given** student has completed the ROS 2 modules, **When** they implement navigation tasks, **Then** they can use Nav2 effectively for autonomous navigation.

2. **Given** student has access to Jetson Orin hardware, **When** they deploy their capstone project, **Then** it runs at ≥15 Hz real-time inference as specified.

---
### User Story 5 - Access Complete Course Materials and Support (Priority: P3)

Instructors and TAs have access to complete course materials including week-by-week syllabus, slide deck outlines, 13+ graded assignment prompts, full grading rubrics, and comprehensive setup tutorials.

**Why this priority**: This enables successful course delivery by educational staff and ensures consistent student experience across different institutions.

**Independent Test**: Instructors can successfully deliver the course using only the provided materials, with TAs able to support students effectively.

**Acceptance Scenarios**:

1. **Given** instructor has access to all course materials, **When** they teach the course, **Then** they can deliver content consistently with the intended learning outcomes.

2. **Given** TA has access to grading rubrics, **When** they evaluate student work, **Then** they can apply consistent evaluation criteria across all assignments.

---

### Edge Cases

- What happens when student has limited access to high-end GPU hardware required for simulation?
- How does the system handle different Ubuntu 22.04 LTS configurations or hardware variations?
- What if specific hardware components become unavailable or discontinued before 2026?
- How does the course handle students with varying levels of Python and deep learning experience beyond the prerequisites?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a complete 13-week syllabus with week-by-week content for Physical AI & Humanoid Robotics capstone course
- **FR-002**: System MUST include slide deck outlines for all 13 weeks of instruction
- **FR-003**: System MUST provide 13+ graded assignment prompts with clear requirements and objectives
- **FR-004**: System MUST include full grading rubrics for all assignments and the capstone project
- **FR-005**: System MUST provide comprehensive setup tutorials for Ubuntu 22.04 LTS environment
- **FR-006**: System MUST enable end-to-end voice-controlled autonomous humanoid pipeline: spoken command → Whisper → LLM planner → ROS 2 action sequence → navigation + manipulation
- **FR-007**: System MUST support deployment in NVIDIA Isaac Sim environment for simulation
- **FR-008**: System MUST support deployment on at least one physical humanoid platform for real-world testing
- **FR-009**: System MUST provide three costed lab tier options with exact part numbers and purchase links: Budget (<$1k), Mid ($3–5k), Premium ($15k+)
- **FR-010**: System MUST include cloud-native fallback path (AWS/NVIDIA Omniverse) with total student cost under $300/quarter
- **FR-011**: System MUST support the modern 2025–2026 embodied-AI stack: ROS 2 Humble/Iron, NVIDIA Isaac Sim + Isaac ROS, Nav2, Vision-Language-Action models
- **FR-012**: System MUST ensure capstone project runs at ≥15 Hz real-time inference on Jetson Orin Nano/Orin NX 8–16 GB
- **FR-013**: System MUST be officially supported in 2026 with active community/package maintenance for all recommended tools
- **FR-014**: System MUST accommodate students with Python and basic deep learning knowledge but no prior ROS or robotics experience
- **FR-015**: System MUST ensure all assignments are doable on student-owned hardware or provided lab kits

### Key Entities

- **Course Package**: Complete educational material including syllabus, slides, assignments, rubrics, and tutorials
- **Student Learning Path**: Progression from zero robotics experience to deploying voice-controlled autonomous humanoid
- **Lab Configuration Options**: Three tiered hardware options (Budget, Mid, Premium) plus cloud fallback
- **Technology Stack**: ROS 2 Humble/Iron, NVIDIA Isaac Sim + Isaac ROS, Nav2, Vision-Language-Action models, Jetson Orin
- **Capstone Project**: End-to-end voice-controlled autonomous humanoid system with specified performance requirements

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Educational institutions can launch the complete 13-week capstone course in Q1 2026 with no missing pieces, enabling students to progress from zero robotics experience to deploying voice-controlled autonomous humanoid systems
- **SC-002**: Students successfully complete the end-to-end capstone demo: spoken command → Whisper → LLM planner → ROS 2 action sequence → navigation + manipulation, validated in both Isaac Sim and on at least one physical platform
- **SC-003**: Course achieves 90% student completion rate across all 13 weeks with students meeting the minimum performance requirement of ≥15 Hz real-time inference on Jetson Orin
- **SC-004**: Three lab tier options are successfully implemented with exact part numbers and purchase links remaining valid in 2025, with cloud fallback keeping total student cost under $300/quarter
- **SC-005**: All recommended tools in the 2025–2026 embodied-AI stack maintain official support and active community/package maintenance through 2026