---
id: "0001"
title: "Humanoid Robotics Course Spec"
stage: spec
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: 001-humanoid-robotics-course
branch: 001-humanoid-robotics-course
user: samiullah
command: /sp.specify
labels: ["specification","course","robotics","ai"]
links:
  spec: "/mnt/d/Quarter-4/spec_kit_plus/humenoid_robot/specs/001-humanoid-robotics-course/spec.md"
  ticket: null
  adr: null
  pr: null
files:
 - /mnt/d/Quarter-4/spec_kit_plus/humenoid_robot/specs/001-humanoid-robotics-course/spec.md
 - /mnt/d/Quarter-4/spec_kit_plus/humenoid_robot/specs/001-humanoid-robotics-course/checklists/requirements.md
tests:
 - unit tests for course materials
 - integration tests for simulation environment
 - performance tests for real-time inference
---

## Prompt

Complete Specification for the "Physical AI & Humanoid Robotics" Capstone Course phase 1 only, dont touch phase 2 in constitution for now.

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

Deliver the complete, ready-to-launch course specification package in Markdown with embedded file structure.

## Response snapshot

Created complete specification for the "Physical AI & Humanoid Robotics" Capstone Course. The specification includes:

- User Scenarios & Testing with 5 prioritized user stories covering course launch, end-to-end system execution, flexible lab setup options, modern AI stack curriculum, and course materials access
- 15 functional requirements covering the complete 13-week course package, end-to-end pipeline, simulation and hardware support, lab tier options, cloud fallback, technology stack, performance requirements, and student prerequisites
- 5 measurable success criteria focused on course launch capability, student completion of end-to-end demo, performance benchmarks, lab option availability, and tool support through 2026
- Key entities including Course Package, Student Learning Path, Lab Configuration Options, Technology Stack, and Capstone Project
- Edge cases for hardware access, system variations, component availability, and student experience levels

Quality checklist created and all validation criteria passed. The specification is ready for the planning phase.

## Outcome

- ✅ Impact: Complete course specification created with all required sections, ready for planning phase
- 🧪 Tests: Specification quality validated against all checklist criteria
- 📁 Files: spec.md and requirements.md created in feature directory
- 🔁 Next prompts: Ready for /sp.plan to develop implementation plan
- 🧠 Reflection: Specification successfully captures all requirements without implementation details

## Evaluation notes (flywheel)

- Failure modes observed: None - specification meets all requirements
- Graders run and results (PASS/FAIL): All validation criteria passed
- Prompt variant (if applicable): Standard specification creation
- Next experiment (smallest change to try): Proceed to planning phase with /sp.plan
