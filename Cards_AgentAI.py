import warnings
warnings.filterwarnings('ignore')

import os
import streamlit as st
import requests
import graphviz
import pandas as pd
import json
import time
from crewai import Agent, Task, Crew, Process
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure LiteLLM
litellm.drop_params = True
litellm.api_timeout = 45
litellm.max_retries = 3
litellm._turn_on_debug()

# Set OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ OPENROUTER_API_KEY not set in environment.")
    st.stop()
os.environ["OPENROUTER_API_KEY"] = api_key

# Verify OpenRouter API Key
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def verify_openrouter_auth():
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://lloyds-bank.com",
            "X-Title": "DebitCardModernizer"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            st.session_state['openrouter_models'] = len(models)
            return True
        st.error(f"❌ OpenRouter auth failed: {response.status_code} - {response.text}")
        return False
    except Exception as e:
        st.error(f"Error verifying OpenRouter key: {str(e)}")
        return False

if not verify_openrouter_auth():
    st.stop()

openrouter_model_id = "openrouter/deepseek/deepseek-r1:free"

DEFAULT_SYSTEM_DESCRIPTION = """CardTransactionPlus (CTP) - Mainframe COBOL Debit Card Processing System

Components:
1. Authorization Engine (COBOL/CICS)
   - Real-time transaction approval
   - Hardcoded business rules (limits, merchant blocks)
   - Visa/MasterCard integration via ISO 8583

2. Account Balancer (COBOL/DB2)
   - Updates customer balances
   - Handles pending/posted states
   - Batch synchronization with core banking

3. Fraud Monitor (COBOL/JCL)
   - Hourly batch jobs
   - Rule-based suspicious activity detection
   - Generates CSV reports

Key Challenges:
- 4-hour batch processing windows
- No real-time fraud detection
- Scaling issues during peak periods
- COBOL specialist dependency
"""

def create_debit_card_agents():
    """Create agents with enhanced configurations"""
    agents = [
        Agent(
            role="Mainframe Payment Systems Architect",
            goal="Analyze COBOL systems and identify modernization opportunities",
            backstory="Senior mainframe expert with 15+ years in payment systems",
            llm=openrouter_model_id,
            verbose=True,
            memory=True,
            max_iter=15,
            max_rpm=10
        ),
        Agent(
            role="Payments Modernization Expert",
            goal="Design cloud-native architectures for payment systems",
            backstory="Former Visa architect specializing in PCI-compliant cloud migrations",
            llm=openrouter_model_id,
            verbose=True,
            memory=True,
            max_iter=15,
            max_rpm=10
        ),
        Agent(
            role="Banking Migration Risk Specialist",
            goal="Identify and mitigate migration risks",
            backstory="CTO with experience in 10+ core banking migrations",
            llm=openrouter_model_id,
            verbose=True,
            memory=True,
            max_iter=15,
            max_rpm=10
        )
    ]
    return agents

def create_debit_card_tasks(system_description: str, agents: list):
    """Create tasks with enhanced instructions"""
    analysis_task = Task(
        description=f"""Analyze this debit card system:
        {system_description}
        
        Provide JSON output with:
        - architecture: {{nodes: [], edges: []}}
        - findings: [] (3-5 key pain points)
        Example: {{"architecture": {{"nodes": ["Auth"], "edges": []}}, "findings": ["Finding1"]}}""",
        expected_output="Valid JSON structure with architecture and findings",
        agent=agents[0],
        output_file="current_state.md"
    )

    design_task = Task(
        description="""Design modern architecture with:
        - services: [{name:, tech:, responsibility:}]
        - compliance: []
        Example: {"services": [{"name": "Auth", "tech": ["Java"]}], "compliance": ["PCI"]}""",
        expected_output="Valid JSON structure with services and compliance",
        agent=agents[1],
        context=[analysis_task],
        output_file="target_architecture.md"
    )

    risk_task = Task(
        description="""Identify risks with:
        - high_risk: []
        - mitigations: []
        Example: {"high_risk": ["Risk1"], "mitigations": ["Mit1"]}""",
        expected_output="Valid JSON structure with risks and mitigations",
        agent=agents[2],
        context=[analysis_task, design_task],
        output_file="risk_assessment.md"
    )
    return [analysis_task, design_task, risk_task]

def extract_structured_output(output: str) -> dict:
    """Robust output parsing with multiple fallback strategies"""
    if isinstance(output, dict):
        return output
        
    try:
        # Try direct JSON parse first
        return json.loads(output)
    except:
        pass
        
    try:
        # Extract JSON from markdown code block
        start = output.find('{')
        end = output.rfind('}') + 1
        if start > -1 and end > 0:
            return json.loads(output[start:end])
    except:
        pass
        
    # Fallback to key-value extraction
    result = {}
    if "findings" in output.lower():
        result["findings"] = [line.strip('- ') for line in output.split('\n') if line.strip().startswith('-')]
    if "services" in output.lower():
        result["services"] = [{"name": line.split(':')[0].strip(), "description": line.split(':')[1].strip()} 
                             for line in output.split('\n') if ':' in line]
    return result

def run_analysis(system_description: str):
    """Execute the full analysis workflow"""
    with st.spinner("Initializing analysis..."):
        agents = create_debit_card_agents()
        tasks = create_debit_card_tasks(system_description, agents)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            manager_llm=openrouter_model_id,
            verbose=True,
            memory=True
        )
        
        try:
            results = crew.kickoff()
            st.session_state.raw_results = results
            
            # Process task outputs
            combined = {}
            for i, task in enumerate(tasks):
                output = getattr(task.output, 'raw_output', str(task.output))
                parsed = extract_structured_output(output)
                
                if i == 0:
                    combined["current_state"] = parsed
                elif i == 1:
                    combined["target_architecture"] = parsed
                elif i == 2:
                    combined["risk_assessment"] = parsed
            
            return combined
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            if 'tasks' in locals():
                for i, task in enumerate(tasks):
                    output = getattr(task.output, 'raw_output', "No output generated")
                    st.text(f"Task {i+1} output:\n{output[:1000]}")
            return None

def show_results(results):
    """Display results in the UI"""
    tab1, tab2, tab3 = st.tabs(["Current State", "Target Architecture", "Risk Assessment"])
    
    with tab1:
        st.subheader("Current System Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Architecture Diagram")
            graph = graphviz.Digraph()
            nodes = results["current_state"].get("architecture", {}).get("nodes", ["Auth", "Balancer", "Fraud"])
            edges = results["current_state"].get("architecture", {}).get("edges", [])
            
            for node in nodes:
                graph.node(node)
            for edge in edges:
                graph.edge(edge[0], edge[1])
            st.graphviz_chart(graph)
        
        with col2:
            st.markdown("#### Key Findings")
            findings = results["current_state"].get("findings", ["No findings generated"])
            for item in findings:
                st.markdown(f"- 🔴 {item}")
    
    with tab2:
        st.subheader("Proposed Architecture")
        
        st.markdown("#### Recommended Services")
        services = results["target_architecture"].get("services", [])
        if services:
            st.dataframe(pd.DataFrame(services), hide_index=True)
        else:
            st.warning("No service recommendations generated")
        
        st.markdown("#### Compliance Requirements")
        for req in results["target_architecture"].get("compliance", ["PCI-DSS"]):
            st.markdown(f"- ✅ {req}")
    
    with tab3:
        st.subheader("Migration Risk Assessment")
        risks = results["risk_assessment"].get("high_risk", ["No risks identified"])
        mitigations = results["risk_assessment"].get("mitigations", ["No mitigations provided"])
        
        risk_df = pd.DataFrame({
            "Risk": risks,
            "Mitigation": mitigations,
            "Severity": ["High"] * len(risks)
        })
        
        st.dataframe(
            risk_df,
            hide_index=True,
            column_config={
                "Severity": st.column_config.SelectboxColumn(
                    options=["Low", "Medium", "High", "Critical"],
                    required=True
                )
            },
            use_container_width=True
        )

def main():
    st.set_page_config(layout="wide", page_title="Debit Card Modernizer", page_icon="💳")
    
    st.title("💳 Debit Card System Modernizer")
    st.caption("For Lloyd's Bank - Prepared for Goldy Samra")
    
    with st.expander("🔍 System Description", expanded=True):
        system_description = st.text_area(
            "Describe your system:",
            height=250,
            value=DEFAULT_SYSTEM_DESCRIPTION,
            label_visibility="collapsed"
        )
    
    if st.button("Analyze System", type="primary"):
        if not system_description.strip():
            st.error("Please enter a system description")
            return
            
        results = run_analysis(system_description)
        if results:
            st.session_state.results = results
            st.success("✅ Analysis completed successfully!")
    
    if 'results' in st.session_state:
        show_results(st.session_state.results)

if __name__ == "__main__":
    main()
