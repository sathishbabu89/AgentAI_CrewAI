# Legacy_System_Analyzer_CrewAI.py

import warnings
warnings.filterwarnings('ignore')

import os
import streamlit as st
import requests
import zipfile
import tempfile
from pathlib import Path
from crewai import Agent, Task, Crew, Process
import litellm

# Enable LiteLLM debug logs
litellm._turn_on_debug()

# Set OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ OPENROUTER_API_KEY not set in environment.")
    raise Exception("OPENROUTER_API_KEY not found.")
os.environ["OPENROUTER_API_KEY"] = api_key

# Verify OpenRouter API Key
def verify_openrouter_auth():
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://your-app-url.com",
            "X-Title": "LegacySystemAnalyzer"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        if response.status_code == 200:
            models = response.json().get("data", [])
            st.session_state['openrouter_models'] = len(models)
            st.success(f"✅ OpenRouter authentication successful! {len(models)} models available.")
        else:
            st.error(f"❌ OpenRouter auth failed: {response.status_code} - {response.text}")
            raise Exception("OpenRouter API key invalid or expired.")
    except Exception as e:
        st.error(f"Error verifying OpenRouter key: {str(e)}")
        raise

verify_openrouter_auth()
openrouter_model_id = "openrouter/deepseek/deepseek-r1:free"

# --- Define Agents ---
code_analyst = Agent(
    role="Code Analyst",
    goal="Understand and summarize the architecture of large legacy codebases",
    backstory="You are a seasoned software analyst with deep experience in dissecting large monolithic codebases and identifying architectural components.",
    verbose=True,
    allow_delegation=False,
    llm=openrouter_model_id
)

data_flow_mapper = Agent(
    role="Data Flow Mapper",
    goal="Map data flows between components and identify data ownership patterns",
    backstory="You specialize in database analysis and data flow tracking in large systems.",
    verbose=True,
    allow_delegation=False,
    llm=openrouter_model_id
)

# Helper function to extract text from repo files
def extract_repo_summary(repo_dir):
    summary = []
    for filepath in Path(repo_dir).rglob("*.*"):
        if filepath.suffix in {".py", ".java", ".js", ".ts", ".go", ".rb", ".cs"}:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                summary.append(f"### File: {filepath.relative_to(repo_dir)}\n```{filepath.suffix[1:]}\n{code[:2000]}\n```\n")
            except Exception as e:
                print(f"Skipping file {filepath}: {e}")
    return "\n".join(summary[:20])  # Limit to 20 files to keep it manageable

# --- Streamlit UI ---
def run_codebase_analyzer():
    st.title("📦 Git Repo Analyzer (CrewAI)")

    uploaded_zip = st.file_uploader("📁 Upload a zipped Git repo:", type=["zip"])

    if uploaded_zip:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "repo.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            repo_summary = extract_repo_summary(temp_dir)
            if not repo_summary:
                st.warning("⚠️ No analyzable code files found in the uploaded ZIP.")
                return

            st.text_area("🧠 Extracted Code Snapshot", value=repo_summary[:5000], height=300)

            if st.button("Analyze Codebase", type="primary"):
                with st.spinner("Analyzing codebase with CrewAI..."):

                    # Task 1 - Codebase Architecture Analysis
                    code_analysis_task = Task(
                        description=(
                            "Analyze the following codebase. Provide a high-level overview of the system architecture, "
                            "major components, dependencies, and potential challenges in modernization:\n\n"
                            f"{repo_summary}"
                        ),
                        expected_output="A structured report with codebase architecture, challenges, and recommendations.",
                        agent=code_analyst,
                    )

                    # Task 2 - Data Flow Mapping
                    data_flow_task = Task(
                        description=(
                            f"""
Based on the codebase architecture described above:
1. Identify which modules own and manage key data
2. Map data flow and data interactions
3. Highlight coupling or tight data dependencies
4. Recommend areas to refactor for microservice decomposition
                        """
                        ),
                        expected_output="A detailed explanation of data ownership and flow in structured text format.",
                        agent=data_flow_mapper,
                        context=[code_analysis_task]
                    )

                    crew = Crew(
                        agents=[code_analyst, data_flow_mapper],
                        tasks=[code_analysis_task, data_flow_task],
                        manager_llm=openrouter_model_id,
                        process=Process.sequential,
                        verbose=True
                    )

                    try:
                        result = crew.kickoff()
                        st.success("✅ Codebase analysis completed!")
                        st.subheader("📊 Final Output:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"❌ Analysis Error: {str(e)}")

# Run app
if __name__ == "__main__":
    run_codebase_analyzer()
