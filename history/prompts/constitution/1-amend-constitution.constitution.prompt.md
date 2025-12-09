---
id: 1
title: Amend Constitution
stage: constitution
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution,setup,initial]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

```text
Project: Phase 1  Constitution for the "Physical AI & Humanoid Robotics" course

Core principles:
- Real-world readiness: every learning objective must map to deployable skills on actual humanoid/embodied platforms
- Hands-on over theory: students must build, simulate, and (when possible) deploy on physical hardware
- Sim-to-Real pipeline: every module must explicitly teach the bridge from simulation (Isaac Sim/Gazebo) → edge device (Jetson) → real robot
- Future-proof stack: prioritize currently industry-dominant tools (ROS 2 Humble/Iron, NVIDIA Isaac Sim/ROS, Nav2, modern VLA approaches)
- Progressive complexity: start simple (single node → full autonomous humanoid with voice commands)

Key standards:
- All tools and versions must be explicitly stated and currently supported (as of 2025)
- Every module must contain at least one graded hands-on assignment that runs on student hardware/cloud
- Capstone project must demonstrate end-to-end autonomy: voice command → LLM planning → ROS 2 action sequence → navigation + manipulation in simulation and (optional) real robot)
- Hardware recommendations must include exact model numbers, minimum specs, and realistic cost tiers
- Safety and ethical deployment of physical robots must be addressed (emergency stops, teleoperation fallback, responsible AI guidelines)

Constraints:
- Course duration: 13 weeks maximum
- Target audience: advanced undergrad or master's students with prior Python + basic ML experience (no prior robotics required)
- Primary OS: Ubuntu 22.04 LTS (dual-boot or native Linux mandatory for frictionless experience)
- GPU requirement: NVIDIA RTX 4070 Ti or higher (or equivalent cloud instance) for Isaac Sim
- Budget transparency: provide three realistic lab tiers (budget < $1k, mid-range $3k–$5k, premium $15k+ per student/group)
- No deprecated tools (e.g., ROS 1, Gazebo Classic, old NVIDIA Jetson TX1/TX2, MoveIt1 without MoveIt2)

Success criteria:
- Students can independently set up a full ROS 2 + Isaac Sim + Jetson development environment from scratch
- Capstone robot (simulated or real) successfully completes: "Pick up the red cup on the table and bring it to me" via natural voice command with zero manual intervention
- 100% of graded assignments run on student-owned or lab-provided hardware (no "lecture-only" modules)
- Course syllabus, weekly schedule, hardware guide, and grading rubric are complete and immediately usable by any university or bootcamp instructor in 2026+
- All recommended hardware kits are purchasable today and supported by manufacturers until at least 2028Project: Phase 2 – Integrated RAG Chatbot Embedded in the "Physical AI & Humanoid Robotics" Book

Core principles:
- Book-native experience: the chatbot must feel like a seamless part of the published digital book (PDF + web version)
- Zero-install for readers: users must be able to talk to the book instantly in-browser with no account creation
- Context-perfect answers: must answer accurately using only the book content + optionally only the user-highlighted text selection
- Production-grade reliability and speed (<1.5 s first token, 99.9% uptime)
- Full cost control: must stay within completely free tiers or <$15/month total operating cost

Key standards:
- RAG pipeline: chunking → embeddings → Qdrant vector store → retrieval → OpenAI gpt-4o-mini or gpt-4.1-mini reasoning
- Backend: FastAPI + Neon Serverless Postgres (for session memory & metadata) + Qdrant Cloud Free Tier (50k vectors)
- Frontend integration: use OpenAI ChatKit / Agents SDK + embedded iframe/widget directly inside the published book (Leanpub, GitBook, and static PDF web viewer)
- User-selected context mode: when user highlights text in the book → "Ask about this selection" button → answer uses only that fragment + relevant retrieved chunks
- All code and infrastructure: 100% reproducible from a single public GitHub repository with docker-compose.yml
- Security: no PII stored, rate-limited per IP, OpenAI API key never exposed client-side

Constraints:
- Vector database: Qdrant Cloud Free Tier only (no paid upgrade)
- Database: Neon Serverless Postgres free tier only
- LLM usage: stay under 2 million tokens/month total (all users combined)
- Deployment platform: Render.com, Fly.io, or Railway free/hobby tier only
- No external authentication required
- Must work on the final published book formats: Leanpub Markdown-rendered web version + static HTML book + PDF with embedded web viewer

Success criteria:
- Any reader can highlight a paragraph in the digital book and ask "Explain this in simpler terms" and receive a perfect, citation-aware answer in <2 seconds
- General questions about the book ("How do I install Isaac ROS on Jetson?") return answers with exact chapter/section references and code snippets
- Live demo runs 24/7 with zero downtime during the entire hackathon judging period
- Entire stack can be deployed from the repo in <10 minutes by any judge
- Total monthly operating cost proven to be ≤$12 (screenshots of billing dashboards required)
- Zero hallucinations when "selected-text-only" mode is used keep both phases apart so we can proceed step by step
```

## Response snapshot

Created comprehensive constitution for the Physical AI & Humanoid Robotics course project, incorporating both Phase 1 (course development) and Phase 2 (integrated RAG chatbot) requirements. The constitution includes:

- Six core principles covering real-world readiness, hands-on learning, sim-to-real pipeline, future-proof stack, progressive complexity, and zero-install book experience
- Detailed standards for both phases including technology stacks, backend/frontend requirements, and integration specifications
- Comprehensive constraints for course duration, target audience, hardware requirements, and deployment limitations
- Clear success criteria for both the educational outcomes and the chatbot functionality
- Development workflow guidelines for both course and chatbot development
- Governance section establishing the constitution as the authoritative guide for all project decisions

The constitution replaces all template placeholders with specific content relevant to the humanoid robotics course and embedded chatbot project, ensuring all development activities align with the defined principles and constraints.

## Outcome

- ✅ Impact: Created foundational constitution document that will guide all future development of the Physical AI & Humanoid Robotics course and integrated chatbot
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with comprehensive content for both phases
- 🔁 Next prompts: Course module development, chatbot implementation, and infrastructure setup based on established principles
- 🧠 Reflection: The constitution successfully captures both educational and technical requirements in a unified document that balances the needs of course development with the innovative chatbot integration.

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin developing Phase 1 course modules based on the established constitution