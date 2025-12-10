<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- Modified principles: Enhanced Phase 1 and Phase 2 principles with detailed content
- Added sections: Comprehensive Key Standards, Constraints, Success Criteria, and Development Workflows for both Phase 1 and Phase 2
- Removed sections: None
- Templates requiring updates: ⚠ pending - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md
- Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics Course Constitution

## Core Principles - Phase 1: Course Development

### I. Real-World Readiness
Every learning objective must map to deployable skills on actual humanoid/embodied platforms. Students must gain hands-on experience with industry-standard tools and platforms that are currently in use in the robotics field. This ensures graduates can immediately contribute to real-world robotics projects.

### II. Hands-On Over Theory
Students must build, simulate, and (when possible) deploy on physical hardware. All modules must include practical implementation components rather than purely theoretical instruction. This approach ensures deep understanding through direct experience with the challenges of physical systems.

### III. Sim-to-Real Pipeline
Every module must explicitly teach the bridge from simulation (Isaac Sim/Gazebo) → edge device (Jetson) → real robot. Students must understand how to develop in simulation and successfully transfer those capabilities to physical hardware, addressing the reality gap between simulated and real environments.

### IV. Future-Proof Stack
Prioritize currently industry-dominant tools including ROS 2 Humble/Iron, NVIDIA Isaac Sim/ROS, Nav2, and modern VLA approaches. All technology choices must be actively supported and have strong industry adoption to ensure long-term relevance of the skills taught.

### V. Progressive Complexity
Start simple (single node → full autonomous humanoid with voice commands). The curriculum must build complexity gradually, allowing students to master foundational concepts before advancing to more sophisticated applications. This ensures comprehensive understanding and reduces learning friction.

## Key Standards - Phase 1

### Technology Standards
- All tools and versions must be explicitly stated and currently supported (as of 2025)
- ROS 2 Humble Hawksbill (or Iron Irwini) as the primary middleware framework
- NVIDIA Isaac Sim as the primary simulation environment
- Isaac ROS perception and navigation packages for perception and control
- Python 3.10+ and C++17 as primary development languages
- Git-based version control with feature branch workflow

### Educational Standards
- Every module must contain at least one graded hands-on assignment that runs on student hardware/cloud
- All code examples must be production-ready with proper error handling and documentation
- Assessment rubrics must be clearly defined with measurable outcomes
- Peer review and code quality standards must be enforced
- Industry mentorship and guest lectures integrated into curriculum

### Infrastructure Standards
- Capstone project must demonstrate end-to-end autonomy: voice command → LLM planning → ROS 2 action sequence → navigation + manipulation in simulation and (optional) real robot
- Hardware recommendations must include exact model numbers, minimum specs, and realistic cost tiers
- Safety and ethical deployment of physical robots must be addressed (emergency stops, teleoperation fallback, responsible AI guidelines)
- Cloud deployment options must be provided for students without high-end hardware
- Continuous integration pipeline for all student projects

### Quality Standards
- Code must follow ROS 2 best practices and style guidelines
- All implementations must include comprehensive unit and integration tests
- Documentation must be maintained at all times with API references
- Performance benchmarks must be established and tracked for all modules

## Constraints - Phase 1

### Time and Duration Constraints
- Course duration: 13 weeks maximum (one academic quarter/semester)
- Weekly commitment: 10-15 hours per week (lectures + labs + assignments)
- Capstone project: 3-4 weeks minimum for full implementation
- Midterm project: 2-3 weeks for foundational skills demonstration

### Technical Constraints
- Target audience: advanced undergrad or master's students with prior Python + basic ML experience (no prior robotics required)
- Primary OS: Ubuntu 22.04 LTS (dual-boot or native Linux mandatory for frictionless experience)
- GPU requirement: NVIDIA RTX 4070 Ti or higher (or equivalent cloud instance) for Isaac Sim
- Minimum RAM: 32GB for optimal simulation performance
- Network requirement: Stable internet for package installation and cloud services

### Budget and Resource Constraints
- Budget transparency: provide three realistic lab tiers (budget < $1k, mid-range $3k–$5k, premium $15k+ per student/group)
- Open-source tools prioritized over commercial solutions where possible
- Cloud computing credits budgeted for student use (AWS Educate, NVIDIA Developer Program)
- Hardware sharing protocols for expensive equipment
- Virtual lab environments as backup for physical hardware limitations

### Technology Constraints
- No deprecated tools (e.g., ROS 1, Gazebo Classic, old NVIDIA Jetson TX1/TX2, MoveIt1 without MoveIt2)
- All software must be compatible with Ubuntu 22.04 LTS
- Hardware must support real-time control requirements
- Simulation fidelity must match physical robot capabilities

## Success Criteria - Phase 1

### Technical Competency Criteria
- Students can independently set up a full ROS 2 + Isaac Sim + Jetson development environment from scratch
- Students demonstrate proficiency with ROS 2 concepts: nodes, topics, services, actions, parameters
- Students implement perception pipelines using Isaac ROS packages or equivalent
- Students develop navigation and manipulation capabilities for humanoid robots
- Students integrate LLMs with ROS 2 for high-level command processing

### Project-Based Criteria
- Capstone robot (simulated or real) successfully completes: "Pick up the red cup on the table and bring it to me" via natural voice command with zero manual intervention
- Students complete a minimum of 5 substantial hands-on projects throughout the course
- All student projects demonstrate proper software engineering practices
- Students implement safety mechanisms and emergency stop procedures
- Final projects include comprehensive documentation and presentation

### Curriculum Completion Criteria
- 100% of graded assignments run on student-owned or lab-provided hardware (no "lecture-only" modules)
- Course syllabus, weekly schedule, hardware guide, and grading rubric are complete and immediately usable by any university or bootcamp instructor in 2026+
- All recommended hardware kits are purchasable today and supported by manufacturers until at least 2028
- Students demonstrate ability to troubleshoot common robotics development issues
- Students complete a professional portfolio of robotics projects

### Industry Readiness Criteria
- Students can contribute to real-world robotics projects immediately after course completion
- Students understand the complete development lifecycle from simulation to deployment
- Students demonstrate knowledge of safety, ethics, and regulatory considerations in robotics
- Students can effectively communicate technical concepts to both technical and non-technical audiences

## Development Workflow - Phase 1

### Course Development Process
- All modules must follow the Sim-to-Real pipeline teaching approach
- Each module requires hands-on assignments with clear grading rubrics
- Hardware requirements must be validated on actual systems before course release
- Safety protocols must be integrated into every practical exercise
- Regular updates to maintain compatibility with evolving robotics frameworks
- Industry advisory board reviews curriculum annually for relevance
- Student feedback incorporated into continuous improvement cycles
- External expert reviews conducted for technical accuracy

## Core Principles - Phase 2: Integrated RAG Chatbot

### VI. Zero-Install Book Experience
For the integrated RAG chatbot, readers must be able to interact with the book instantly in-browser with no account creation. The chatbot must feel like a seamless part of the published digital book across PDF and web formats, providing immediate access to knowledge without barriers.

### VII. Context-Aware Intelligence
The chatbot must understand and maintain context from the book content, providing answers that are directly relevant to the specific text and concepts covered in the course materials.

### VIII. Citation-First Accuracy
All responses must be grounded in the book content with proper citations and references, ensuring zero hallucinations and maintaining academic integrity.

### IX. Performance-First Design
The system must prioritize fast response times (<2 seconds) and high availability (99.5% uptime) to ensure a seamless reading experience that doesn't interrupt the learning flow.

## Key Standards - Phase 2

### RAG Pipeline Standards
- RAG pipeline: chunking → embeddings → Qdrant vector store → retrieval → OpenAI gpt-4o-mini or gpt-4.1-mini reasoning
- Text chunking: semantic boundaries with 50-200 word overlapping windows
- Embedding model: OpenAI ada-002 or equivalent for optimal retrieval
- Retrieval: top-3 most relevant chunks with relevance scoring
- Response generation: citation-aware with source attribution

### Backend Architecture Standards
- Backend: FastAPI + Neon Serverless Postgres (for session memory & metadata) + Qdrant Cloud Free Tier (50k vectors)
- API rate limiting: 10 requests per minute per IP to stay within free tier
- Session management: temporary storage with 24-hour expiration
- Error handling: comprehensive logging and graceful degradation
- Security: input sanitization and output encoding to prevent injection attacks

### Frontend Integration Standards
- Frontend integration: use OpenAI ChatKit / Agents SDK + embedded iframe/widget directly inside the published book (Leanpub, GitBook, and static PDF web viewer)
- Responsive design: works on mobile, tablet, and desktop devices
- Accessibility: WCAG 2.1 AA compliance for inclusive access
- Performance: <2 second response times for all user interactions
- Offline capability: cached content for basic functionality

### User Experience Standards
- User-selected context mode: when user highlights text in the book → "Ask about this selection" button → answer uses only that fragment + relevant retrieved chunks
- Natural language interface with intelligent query understanding
- Context-aware responses that maintain conversation history
- Multi-modal support for text and image-based queries
- Personalization: remembers user preferences and learning patterns

### Quality and Reliability Standards
- All code and infrastructure: 100% reproducible from a single public GitHub repository with docker-compose.yml
- Security: no PII stored, rate-limited per IP, OpenAI API key never exposed client-side
- Comprehensive testing: unit, integration, and end-to-end tests
- Monitoring: performance metrics and error tracking
- Backup and recovery: automated backups of all non-transient data

## Constraints - Phase 2

### Infrastructure Constraints
- Vector database: Qdrant Cloud Free Tier only (no paid upgrade)
- Database: Neon Serverless Postgres free tier only
- LLM usage: stay under 2 million tokens/month total (all users combined)
- Deployment platform: Render.com, Fly.io, or Railway free/hobby tier only
- No external authentication required
- Must work on the final published book formats: Leanpub Markdown-rendered web version + static HTML book + PDF with embedded web viewer

### Technical Constraints
- Maximum 50,000 vectors in Qdrant Cloud Free Tier
- 10,000 rows in Neon Serverless Postgres free tier
- 500GB/month bandwidth limit on hosting platforms
- 512MB RAM limit on free tier deployments
- 100 concurrent connections maximum

### Performance Constraints
- Response time: <2 seconds for 95% of requests
- Uptime: 99.5% availability during peak hours
- Concurrent users: support up to 50 simultaneous users
- File upload: maximum 10MB per document
- Session duration: maximum 24 hours per session

### Cost Constraints
- Monthly operating costs: ≤$12 total across all services
- No premium features that require paid upgrades
- Optimized resource usage to maximize free tier benefits
- Cost monitoring and alerting implemented
- Usage analytics to predict and prevent overages

## Success Criteria - Phase 2

### Functional Success Criteria
- Any reader can highlight a paragraph in the digital book and ask "Explain this in simpler terms" and receive a perfect, citation-aware answer in <2 seconds
- General questions about the book ("How do I install Isaac ROS on Jetson?") return answers with exact chapter/section references and code snippets
- Multi-modal queries: users can ask questions about diagrams, code samples, and images in the book
- Context-aware responses: system remembers conversation history within session
- Code examples: can extract and explain specific code snippets from the book

### Performance Success Criteria
- Live demo runs 24/7 with zero downtime during the entire hackathon judging period
- Entire stack can be deployed from the repo in <10 minutes by any judge
- Total monthly operating cost proven to be ≤$12 (screenshots of billing dashboards required)
- Zero hallucinations when "selected-text-only" mode is used
- 95th percentile response time <2 seconds
- 99.5% uptime maintained over 30-day period

### Quality Success Criteria
- Accuracy: 98% of responses are factually correct and cite appropriate sources
- Relevance: 95% of responses directly address the user's question
- Completeness: responses provide sufficient context and detail
- Consistency: similar questions receive consistent answers
- Safety: 100% of responses adhere to responsible AI guidelines

### User Experience Success Criteria
- Usability: 90% of users can complete their first query within 2 minutes
- Satisfaction: 4.5/5 star rating from user feedback
- Engagement: average session duration >5 minutes
- Adoption: 80% of book readers use the chat feature at least once
- Accessibility: passes automated accessibility testing

## Development Workflow

### Course Development Process:
- All modules must follow the Sim-to-Real pipeline teaching approach
- Each module requires hands-on assignments with clear grading rubrics
- Hardware requirements must be validated on actual systems before course release
- Safety protocols must be integrated into every practical exercise
- Regular updates to maintain compatibility with evolving robotics frameworks

### Chatbot Integration Process:
- RAG pipeline must be tested with full book content before deployment
- Response accuracy validated against book content with no hallucinations
- Performance benchmarks must meet <1.5s response time requirements
- Cost monitoring implemented to ensure free tier usage limits
- Security review completed to ensure no PII collection or exposure
- A/B testing framework for continuous improvement
- User feedback integration for model refinement

## Governance

This constitution supersedes all other course development and chatbot implementation practices. All curriculum and technical decisions must align with the stated principles and constraints. Amendments to this constitution require explicit documentation of the change, approval from project stakeholders, and a migration plan for any affected course materials or code implementations. All pull requests and code reviews must verify compliance with both Phase 1 and Phase 2 requirements. Complexity must be justified by clear educational or technical benefits. Use this constitution as the primary guidance document for all development decisions.

**Version**: 1.1.0 | **Ratified**: 2025-12-09 | **Last Amended**: 2025-12-09
