# Legacy_System_Analyzer_CrewAI.py

import warnings
warnings.filterwarnings('ignore')

import os
import streamlit as st
import requests
from crewai import Agent, Task, Crew, Process
import litellm

# Enable LiteLLM debug logs
litellm._turn_on_debug()

# Set OpenRouter API key (ensure it is set securely as an environment variable)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ OPENROUTER_API_KEY not set in environment.")
    raise Exception("OPENROUTER_API_KEY not found.")

os.environ["OPENROUTER_API_KEY"] = api_key  # Make sure LiteLLM picks it up

# ✅ Verify OpenRouter API Key
def verify_openrouter_auth():
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://your-app-url.com",  # Replace with your app domain if you have one
            "X-Title": "LegacySystemAnalyzer"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        if response.status_code == 200:
            models = response.json().get("data", [])
            st.session_state['openrouter_models'] = len(models)
            st.success(f"✅ OpenRouter authentication successful! {len(models)} models available.")
            print("OpenRouter authentication successful!")
        else:
            st.error(f"❌ OpenRouter auth failed: {response.status_code} - {response.text}")
            raise Exception("OpenRouter API key invalid or expired.")
    except Exception as e:
        st.error(f"Error verifying OpenRouter key: {str(e)}")
        raise

# Call this before setting up CrewAI agents
verify_openrouter_auth()

# Define OpenRouter model
openrouter_model_id = "openrouter/deepseek/deepseek-r1:free"

# Define Agent using OpenRouter model
enterprise_architect = Agent(
    role="Enterprise Architect",
    goal="Analyze and evaluate legacy banking systems for modernization opportunities",
    backstory=(
        "You're an expert enterprise architect with 20+ years of experience in large-scale banking systems. "
        "You understand technical debt, monolithic architectures, and modern cloud-native transformation."
    ),
    verbose=True,
    allow_delegation=False,
    llm=openrouter_model_id  # Pass OpenRouter model ID directly
)

# Create analysis task
def create_analysis_task(system_description: str):
    return Task(
        description=(
            "Based on the following system description, analyze the legacy banking system. "
            "Provide an overview of the architecture, highlight key challenges, and suggest potential modernization strategies:\n\n"
            f"{system_description}"
        ),
        expected_output="A structured report with architectural overview, challenges, and modernization roadmap.",
        agent=enterprise_architect,
    )

# Streamlit UI
def run_legacy_analyzer():
    st.title("Legacy Banking System Analyzer (CrewAI)")

    system_description = st.text_area(
        "Describe your banking system:",
        height=200,
        value="""Monolithic banking application with:
- CustomerManagement (500K LOC)
- AccountServices (800K LOC) 
- LoanProcessing (600K LOC)
- Shared Oracle database"""
    )

    if st.button("Analyze", type="primary"):
        if system_description.strip():
            with st.spinner("Analyzing system with CrewAI..."):
                analysis_task = create_analysis_task(system_description)

                crew = Crew(
                    agents=[enterprise_architect],
                    tasks=[analysis_task],
                    manager_llm=openrouter_model_id,  # Manager uses same model
                    process=Process.sequential,
                    verbose=True
                )

                try:
                    result = crew.kickoff()
                    st.success("✅ Analysis completed!")
                    st.subheader("📋 System Analysis:")
                    st.write(result)
                except Exception as e:
                    st.error(f"❌ Analysis Error: {str(e)}")
        else:
            st.error("Please enter a system description")

# Run app
if __name__ == "__main__":
    run_legacy_analyzer()
