# GitHub Copilot Instructions

## Project Overview

This repository is a documentation resource for Python-based computational research projects. It provides an incomplete but growing guide for building and maintaining Python research codebases from both a research scientist's and programmer's perspective.

Copilot should prioritize:
- Clear, accessible technical writing
- Consistent formatting and structure across documents
- Practical, actionable guidance with examples
- Maintaining the educational tone and research-oriented focus

## Project Structure

- `docs/`: Main documentation files (numbered for reading order)
  - `01_Coding_Convention.md`: Python coding standards
  - `02_Project_Structure.md`: Repository organization
  - `03_Version_Contorl_and_Git.md`: Version control practices
  - `04_Python_Env.md`: Environment and dependency management
  - `05_Config.md`: Configuration management
  - `06_Logging.md`: Logging best practices
  - `07_Cross_Platform.md`: Cross-platform development
  - `08_Remote_Machine.md`: Remote deployment
  - `09_Things_Should_Consdier.md`: General considerations
  - `10_Profiing_and_Speed.md`: Performance optimization
  - `11_Testing.md`: Testing strategies
  - `12_Go_With_Large_Scale.md`: Scaling research code
  - `13_Use_Copilot.md`: AI-assisted development
- `examples/`: Code examples, notebooks, and scripts
- `README.md`: Repository overview and navigation

## Documentation Standards

### Writing Style
- Write in clear, concise English; assume the reader is a researcher or scientist, not necessarily a professional software engineer
- Use an educational, helpful tone
- Provide concrete examples and code snippets where applicable
- Explain the "why" behind recommendations, not just the "what"
- Avoid jargon without explanation; define technical terms when first used

### Formatting Conventions
- Use ATX-style headers (`#`, `##`, `###`) consistently
- Keep headers descriptive and hierarchical
- Use code blocks with language specifiers (```python, ```bash, etc.)
- Use inline code formatting for commands, file names, and code elements
- Use bullet points for lists; numbered lists for sequential steps
- Include blank lines around headers, code blocks, and sections for readability

### Content Organization
- Start each document with a brief overview/introduction
- Use clear section headings to break up content
- Include practical examples after theoretical explanations
- Add cross-references to related documents when relevant
- Keep documents focused on a single topic

### Code Examples
- Provide working, runnable examples when possible
- Use realistic variable/function names (avoid foo/bar)
- Include comments for complex logic
- Show both good and bad examples when illustrating concepts
- Prefer Python 3.10+ syntax and features

## Important Constraints

- Do not change the numbered prefix of document files (01_, 02_, etc.) without explicit instruction
- Maintain consistency with existing document structure and tone
- Do not remove or significantly alter existing examples without preserving their intent
- Keep the research/academic context in mind; avoid over-engineering
- Preserve cross-references between documents when editing

## How Copilot Should Work

- Before editing documentation, read surrounding sections to match tone and style
- When adding new content:
  - Follow the formatting conventions established in similar documents
  - Add concrete Python examples where appropriate
  - Cross-reference related topics in other documents
  - Maintain the balance between accessibility and technical depth
- When suggesting improvements:
  - Prioritize clarity and usefulness for researchers
  - Consider both novice and experienced Python users
  - Provide context and rationale for recommendations
- If creating new documents, follow the existing numbering and naming conventions
