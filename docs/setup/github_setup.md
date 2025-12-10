# GitHub Repository Setup and Branching Strategy

This document outlines the steps to set up the GitHub repository with an appropriate branching strategy for the Physical AI & Humanoid Robotics course.

## Repository Initialization

### 1. Create GitHub Repository

First, create a new repository on GitHub:

1. Go to [GitHub.com](https://github.com)
2. Click the "+" icon in the top-right corner and select "New repository"
3. Fill in the repository details:
   - Repository name: `physical-ai-humanoid-robotics`
   - Description: "Course materials for Physical AI & Humanoid Robotics: From Simulated Brains to Walking Bodies"
   - Visibility: Public (for educational purposes)
   - Initialize with README: Yes
   - Add .gitignore: Select "Python"
   - Add license: Select "Apache 2.0 License"

### 2. Clone Repository Locally

Clone the repository to your local development environment:

```bash
# Clone the repository
git clone https://github.com/your-organization/physical-ai-humanoid-robotics.git
cd physical-ai-humanoid-robotics

# Set up Git configuration
git config core.autocrlf input  # For cross-platform compatibility
git config core.filemode false  # For consistent file permissions
git config core.precomposeunicode true  # For consistent Unicode handling
```

## Branching Strategy

### 1. Main Branches

The repository will use a Git Flow-inspired branching strategy:

- `main` (or `master`): Production-ready code, stable releases
- `develop`: Integration branch for features in development
- `release/*`: Release preparation branches
- `feature/*`: Individual feature development branches
- `hotfix/*`: Critical bug fixes for production
- `docs/*`: Documentation updates

### 2. Branch Creation and Naming Convention

```bash
# Set up the main branches
git checkout -b develop

# Push main and develop branches to remote
git push -u origin main
git push -u origin develop
```

## Repository Structure

### 1. Directory Structure

Create the following directory structure to organize course materials:

```bash
# Create main directory structure
mkdir -p docs/{setup,tutorials,exercises,projects}
mkdir -p src/{simulation,control,perception,navigation}
mkdir -p assets/{images,videos,models}
mkdir -p tests
mkdir -p scripts
mkdir -p notebooks
mkdir -p config

# Create additional structure for the course
mkdir -p course_materials/{week1,week2,week3,week4,week5,week6,week7,week8,week9,week10,week11,week12,week13}
mkdir -p course_materials/projects/{hw1,hw2,hw3,hw4,midterm,final_project}
mkdir -p course_materials/solutions
mkdir -p course_materials/datasets
```

### 2. Initialize with Course-Specific Files

Create initial course-specific files:

```bash
# Create main README.md
cat << 'EOF' > README.md
# Physical AI & Humanoid Robotics: From Simulated Brains to Walking Bodies

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-blue)](https://docs.ros.org/en/jazzy/index.html)
[![Isaac%20Sim](https://img.shields.io/badge/Isaac%20Sim-2024.2+-orange)](https://developer.nvidia.com/isaac-sim)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)

This repository contains course materials for the Physical AI & Humanoid Robotics course, covering topics from simulated brains to walking bodies.

## Course Overview

This comprehensive textbook and lab manual targets university students and industry engineers, providing a complete 13-week capstone course in advanced robotics.

## Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Jazzy (or Iron)
- Isaac Sim 2024.2+
- Isaac ROS 3.0+
- Jetson Orin Nano (for hardware deployment)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/physical-ai-humanoid-robotics.git
   cd physical-ai-humanoid-robotics
   ```

2. Follow the setup guide in [docs/setup/](docs/setup/)

## Course Structure

- **Week 1-3**: Introduction to Physical AI and ROS 2 fundamentals
- **Week 4-6**: Simulation and Isaac Sim integration
- **Week 7-9**: Perception and AI integration
- **Week 10-12**: Control systems and locomotion
- **Week 13**: Capstone project integration

## Contributing

We welcome contributions to improve the course materials. Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
EOF

# Create CONTRIBUTING.md
cat << 'EOF' > CONTRIBUTING.md
# Contributing to Physical AI & Humanoid Robotics Course

We welcome contributions to improve the Physical AI & Humanoid Robotics course materials. This document provides guidelines for contributing.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Coding Standards](#coding-standards)
4. [Documentation Guidelines](#documentation-guidelines)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/physical-ai-humanoid-robotics.git
   cd physical-ai-humanoid-robotics
   ```
3. Set up your development environment following the setup guides in `docs/setup/`

## Development Workflow

We follow a Git Flow-inspired workflow:

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
3. Commit your changes with descriptive commit messages
4. Push your branch to GitHub
5. Create a pull request to merge into `develop`

### Branch Naming Convention

- Feature branches: `feature/short-description`
- Bug fixes: `fix/issue-description`
- Documentation: `docs/update-description`
- Release preparation: `release/vX.Y.Z`

## Coding Standards

### Python
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions, classes, and modules
- Use meaningful variable and function names

### ROS 2
- Follow ROS 2 style guidelines
- Use appropriate message types
- Implement proper error handling
- Follow package naming conventions

## Documentation Guidelines

- Use Markdown for documentation
- Include examples and code snippets
- Provide clear explanations of concepts
- Include visual aids where helpful

## Testing

All contributions should include appropriate tests:

1. Unit tests for Python functions
2. Integration tests for ROS 2 nodes
3. Documentation of test procedures

## Submitting Changes

1. Ensure all tests pass
2. Update documentation as needed
3. Submit a pull request with a clear description
4. Link to any relevant issues
5. Request review from maintainers

### Pull Request Template

When creating a pull request, please use the following template:

```
## Description
Brief description of changes made.

## Type of Change
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] Enhancement (improving existing functionality)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New functionality tested
- [ ] Existing functionality still works

## Checklist
- [ ] Code follows project guidelines
- [ ] Self-review completed
- [ ] Documentation updated if needed
```

## Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers.
EOF

# Create CODE_OF_CONDUCT.md
cat << 'EOF' > CODE_OF_CONDUCT.md
# Code of Conduct

## Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a professional setting

## Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.

## Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event. Representation of a project may be further defined and clarified by project maintainers.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident. Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4, available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
EOF

# Create .gitignore for robotics project
cat << 'EOF' > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg

# C extensions
*.c

# Distribution / packaging
.Python
env/
venv/
ENV/
.venv/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Spyder project settings
.spyproject
.spyproject.*

# Rope project settings
.ropeproject

# VS Code
.vscode/

# PyCharm
.idea/

# Sublime Text
*.sublime-project
*.sublime-workspace

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# ROS specific
*.log
*.bag
*.bag.active
*.log.*

# Simulation specific
*.urdf
*.sdf
*.world
*.dae
*.obj
*.stl
*.fbx
*.blend

# Isaac Sim
isaac_sim/
isaac-sim/
--/isaac-sim/
--/isaac_sim/

# Robot models
models/
urdf/
meshes/

# Build directories
build/
install/
log/

# Data files (large files)
*.csv
*.json
*.xml
*.yaml
*.yml

# Documentation builds
docs/_build/
*.html

# Environment variables
.env
*.env

# Compiled source
*.com
*.class
*.dll
*.exe
*.o
*.so

# Packages
*.tar.gz
*.zip
*.7z

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Environment specific files
.env.local
.env.development.local
.env.test.local
.env.production.local

# ROS 2 specific
*.db3
*.db3-shm
*.db3-wal

# Custom for robotics project
*.gazebo
.gazebo/
~/.gazebo/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/
.mypy_cache/
.pytest_cache/
.hypothesis/

# Ignore large binary files
*.bin
*.dat
*.raw
*.tiff
*.tif
*.bmp
*.ico
*.psd
*.ai
*.eps
*.ps

# Ignore specific directories with large files
large_data/
datasets/
training_data/
simulation_snapshots/
EOF
```

## Git Configuration and Hooks

### 1. Configure Git Hooks

Set up Git hooks to ensure code quality:

```bash
# Create hooks directory
mkdir -p .githooks

# Create a pre-commit hook for basic checks
cat << 'EOF' > .githooks/pre-commit
#!/bin/bash
# Pre-commit hook for robotics project

# Check for large files (>50MB)
echo "Checking for large files..."
git diff --cached --name-only -z | xargs -0 -I {} bash -c 'if [ -f "{}" ] && [ $(stat -c%s "{}") -gt 52428800 ]; then echo "ERROR: File {} is larger than 50MB"; exit 1; fi'

# Check for secrets in staged files
echo "Checking for secrets..."
git diff --cached --name-only | xargs grep -l "password\|secret\|token\|key\|credential" && echo "WARNING: Potential secrets found in staged files" && exit 1

echo "Pre-commit checks passed!"
EOF

chmod +x .githooks/pre-commit

# Install the hooks
git config core.hooksPath .githooks
```

### 2. Set Up Git Aliases

Configure useful Git aliases for the project:

```bash
# Set up Git aliases for common operations
git config alias.st status
git config alias.co checkout
git config alias.br branch
git config alias.ci commit
git config alias.unstage 'reset HEAD --'
git config alias.last 'log -1 HEAD'
git config alias.visual '!gitk'
git config alias.graph 'log --oneline --graph --all'
git config alias.cleanup '!git clean -fd && git remote update && git fetch --all && git reset --hard origin/develop'

# Aliases specific to the branching strategy
git config alias.dev-checkout '!git checkout develop && git pull origin develop'
git config alias.main-checkout '!git checkout main && git pull origin main'
git config alias.create-feature '!f() { git checkout develop && git pull origin develop && git checkout -b feature/$1; }; f'
git config alias.create-fix '!f() { git checkout main && git pull origin main && git checkout -b hotfix/$1; }; f'
```

## Branch Protection Rules

### 1. Set Up Branch Protection (GitHub UI)

Configure branch protection rules in the GitHub repository settings:

1. Go to your repository on GitHub
2. Navigate to Settings > Branches
3. Add branch protection rules for `main`:

```
Branch name pattern: main
✓ Require pull request reviews before merging
  - Required number of reviewers: 1
  - Require review from Code Owners: unchecked
  - Require approval of the most recent reviewable push: checked
  - Dismiss stale pull request approvals when new commits are pushed: checked
  - Require review from specific reviewers: none

✓ Require status checks to pass before merging
  - Require branches to be up to date before merging: checked
  - Status checks that are required: none (or configure specific CI checks)

✓ Require conversation resolution before merging
✓ Restrict who can push to matching branches
  - Allow specified actors to bypass required pull requests: maintainers
```

2. Add branch protection rules for `develop`:

```
Branch name pattern: develop
✓ Require pull request reviews before merging
  - Required number of reviewers: 1

✓ Require status checks to pass before merging
  - Require branches to be up to date before merging: checked

✓ Require conversation resolution before merging
```

## GitHub Actions Setup

### 1. Create GitHub Actions Workflows

Set up CI/CD workflows for the repository:

```bash
# Create directory for GitHub Actions
mkdir -p .github/workflows

# Create a basic CI workflow
cat << 'EOF' > .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ develop, main ]
  pull_request:
    branches: [ develop ]

jobs:
  test:
    runs-on: ubuntu-22.04

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Test with pytest
      run: |
        pytest
EOF

# Create a documentation workflow
cat << 'EOF' > .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [ main, develop ]

jobs:
  build-docs:
    runs-on: ubuntu-22.04

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install documentation dependencies
      run: |
        pip install sphinx sphinx-rtd-theme
        pip install -r docs/requirements.txt || echo "No docs requirements file"

    - name: Build documentation
      run: |
        cd docs
        make html || sphinx-build -b html . _build/html

    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
EOF
```

## Release Strategy

### 1. Semantic Versioning

The project will follow semantic versioning (SemVer):

- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### 2. Release Process

```bash
# Create a release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# Update version numbers in relevant files
# Update changelog
# Run final tests

# Merge to main
git checkout main
git pull origin main
git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge --no-ff release/v1.0.0
git push origin develop

# Delete release branch
git branch -d release/v1.0.0
```

## Repository Settings

### 1. Configure Repository Settings

In the GitHub repository settings:

1. **Options**:
   - Repository name: `physical-ai-humanoid-robotics`
   - Description: Course materials for Physical AI & Humanoid Robotics
   - Website: Link to course website if available
   - Features: Enable Issues, Wiki, Projects as needed

2. **Branches**:
   - Default branch: `main`
   - Set up branch protection as described above

3. **Manage Access**:
   - Add collaborators with appropriate permissions
   - Set up teams if working with multiple instructors/TA's

### 2. Issue Templates

Create issue templates to standardize reporting:

```bash
# Create issue templates directory
mkdir -p .github/ISSUE_TEMPLATE

# Create a bug report template
cat << 'EOF' > .github/ISSUE_TEMPLATE/bug_report.md
---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
- OS: [e.g. Ubuntu 22.04]
- ROS 2 Version: [e.g. Jazzy]
- Isaac Sim Version: [e.g. 2024.2]
- Python Version: [e.g. 3.10]

**Additional context**
Add any other context about the problem here.
EOF

# Create a feature request template
cat << 'EOF' > .github/ISSUE_TEMPLATE/feature_request.md
---
name: Feature request
about: Suggest an idea for this project
title: ''
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
EOF

# Create a course content template
cat << 'EOF' > .github/ISSUE_TEMPLATE/course_content.md
---
name: Course Content Issue
about: Report issues with course materials or suggest improvements
title: ''
labels: documentation, course-content
assignees: ''
---

**Chapter/Section**: [e.g., Chapter 3, Week 5]

**Issue Type**:
- [ ] Typo/Error in content
- [ ] Missing content
- [ ] Unclear explanation
- [ ] Code example not working
- [ ] Suggestion for improvement

**Description**
Please describe the issue or suggestion in detail:

**Suggested Resolution** (if applicable)
A clear and concise description of what should be changed or added:

**Additional context**
Add any other context about the issue here.
EOF
```

## Initial Commit and Push

### 1. Make Initial Commit

```bash
# Add all files
git add .

# Make initial commit
git commit -m "feat: initialize Physical AI & Humanoid Robotics course repository

- Set up directory structure for course materials
- Add initial documentation files (README, CONTRIBUTING, CODE_OF_CONDUCT)
- Configure .gitignore for robotics project
- Set up Git hooks and aliases
- Create GitHub Actions workflows
- Add issue templates

This establishes the foundation for the 13-week capstone course in Physical AI & Humanoid Robotics."

# Push all branches to remote
git push -u origin main
git push -u origin develop
```

## Team Collaboration Setup

### 1. Set Up Code Owners (Optional)

Create a CODEOWNERS file to designate code reviewers:

```bash
cat << 'EOF' > .github/CODEOWNERS
# Code owners for the Physical AI & Humanoid Robotics course

# Course maintainers
* @maintainer1 @maintainer2

# Specific directories
docs/ @documentation-team
src/simulation/ @simulation-team
src/control/ @control-team
src/perception/ @perception-team
src/navigation/ @navigation-team
course_materials/ @course-team
EOF
```

### 2. Set Up Project Board

Create GitHub project boards for tracking:

1. Course Development Board: Track development of course materials
2. Student Issues Board: Track student-reported issues
3. Feature Requests Board: Track enhancement requests

## Security Considerations

### 1. Security Policy

Create a security policy for the repository:

```bash
cat << 'EOF' > SECURITY.md
# Security Policy

## Supported Versions

The following versions of the Physical AI & Humanoid Robotics course materials are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in the course materials, please report it to us responsibly.

**Do not** create a public GitHub issue for security vulnerabilities.

### How to Report

Please email security reports to [security-contact@university.edu] with the following information:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested remediation (if any)

### Response Timeline

We will:
- Acknowledge receipt of your report within 48 hours
- Provide regular updates on the status of your report
- Notify you when the vulnerability has been addressed

### Security Updates

Security updates will be released as soon as possible after a vulnerability is confirmed. We will provide appropriate credit to reporters who follow responsible disclosure practices.
EOF
```

## Next Steps

After setting up the GitHub repository with appropriate branching strategy:

1. The repository is now ready for development
2. Team members can begin creating feature branches from `develop`
3. Follow the established branching strategy for all development work
4. Use the configured GitHub Actions for CI/CD
5. Report issues using the standardized templates

## References

- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Best Practices](https://github.com/topics/best-practices)