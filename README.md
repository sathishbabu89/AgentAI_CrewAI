# AgentAI_CrewAI

Here's a comprehensive `README.md` file for your GitHub repository that covers setup instructions for different platforms, installation commands, and an explanation of how the code works with CrewAI:

```markdown
# Banking Legacy System Decomposition Analyzer

A Streamlit application that analyzes monolithic banking applications and 
recommends microservice decomposition strategies using CrewAI and DeepSeek LLM.

## Features

- Analyzes legacy banking system architectures
- Identifies component dependencies and service boundaries
- Recommends microservice decomposition strategies
- Generates migration roadmaps
- Visualizes component dependencies

## Prerequisites

- Python 3.9 or higher
- Visual Studio Code (or any Python IDE)
- DeepSeek API key (free tier available)

## Setup Instructions

### For Windows Machines (VS Code)

1. **Install Python**:
   - Download Python from [python.org](https://www.python.org/downloads/windows/)
   - Check "Add Python to PATH" during installation

2. **Set up VS Code**:
   - Install VS Code from [code.visualstudio.com](https://code.visualstudio.com/download)
   - Install the Python extension from the marketplace

3. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/banking-legacy-analyzer.git
   cd banking-legacy-analyzer
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

### For Apple Machines (VS Code)

1. **Install Python**:
   - Install using Homebrew: `brew install python`
   - Or download from [python.org](https://www.python.org/downloads/macos/)

2. **Set up VS Code**:
   - Install VS Code from [code.visualstudio.com](https://code.visualstudio.com/download)
   - Install the Python extension

3. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/banking-legacy-analyzer.git
   cd banking-legacy-analyzer
   ```

4. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

## Installation

Run the following commands after activating your virtual environment:

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install streamlit crewai langchain-openai openai networkx matplotlib requests litellm
```
Recommended Python Version:
Python 3.9 or higher is recommended. You can add this to your README's prerequisites section.

To generate/update the requirements file:
If you need to update this later, you can generate it from your virtual environment using:

```bash
pip freeze > requirements.txt
```
This requirements file ensures all users will have the exact versions of packages that were tested with your application, preventing version conflicts. 
The versions specified are current as of June 2024 and are known to work well together.

## How It Works (CrewAI Implementation)

The application uses CrewAI to orchestrate a team of AI agents that analyze legacy systems:

1. **Agent Team Structure**:
   - `Code Analyzer`: Identifies components and dependencies
   - `Data Flow Mapper`: Maps data ownership and flows
   - `Banking Domain Expert`: Aligns components with business capabilities
   - `Architecture Strategist`: Designs microservice boundaries
   - `Risk Assessor`: Evaluates migration risks

2. **Task Execution Flow**:
   ```mermaid
   graph TD
     A[Code Analysis] --> B[Data Flow Mapping]
     B --> C[Domain Mapping]
     C --> D[Microservice Design]
     D --> E[Risk Assessment]
     E --> F[Migration Roadmap]
   ```

3. **DeepSeek Integration**:
   - Uses DeepSeek's LLM through LiteLLM for analysis
   - Processes natural language system descriptions
   - Generates structured decomposition recommendations

## Running the Application

1. **Set your API key**:
   - Create a `.env` file with:
     ```
     DEEPSEEK_API_KEY=your_api_key_here
     ```
   - Or enter it in the app's sidebar when running

2. **Start the application**:
   ```bash
   streamlit run app.py
   ```

3. **Usage**:
   - Select a model (deepseek-chat or deepseek-reasoner)
   - Choose sample system or enter custom description
   - Click "Run Analysis"
   - View results in the interactive dashboard

## Configuration Options

- Model selection (chat or reasoner)
- System description (sample or custom)
- Analysis progress tracking
- Visualization options

## Troubleshooting

If you encounter errors:
1. Verify your DeepSeek API key is correct
2. Check your internet connection
3. Ensure all dependencies are installed
4. Try reducing the complexity of your system description

## License

MIT License - See [LICENSE](LICENSE) for details
```

This README includes:

1. **Setup instructions** for both Windows and macOS with VS Code
2. **Complete pip installation** commands (both via requirements.txt and manual)
3. **Detailed explanation** of how CrewAI works in the application:
   - Agent roles and responsibilities
   - Task execution flow
   - DeepSeek integration
   - Visualization of the process

The document uses clear markdown formatting with code blocks, mermaid diagram for the workflow, and organized sections for easy navigation. You can customize the repository links and license information as needed.
