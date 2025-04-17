import warnings
warnings.filterwarnings('ignore')

import os
import streamlit as st
import requests
import graphviz
import pandas as pd
from crewai import Agent, Task, Crew, Process
import litellm
from typing import Dict, Any

# Enable LiteLLM debug logs
litellm._turn_on_debug()

# Set OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ OPENROUTER_API_KEY not set in environment.")
    st.stop()
os.environ["OPENROUTER_API_KEY"] = api_key

# Verify OpenRouter API Key
def verify_openrouter_auth():
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://lloyds-bank.com",
            "X-Title": "DebitCardModernizer"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        if response.status_code == 200:
            models = response.json().get("data", [])
            st.session_state['openrouter_models'] = len(models)
            return True
        else:
            st.error(f"❌ OpenRouter auth failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Error verifying OpenRouter key: {str(e)}")
        return False

if not verify_openrouter_auth():
    st.stop()

openrouter_model_id = "openrouter/deepseek/deepseek-r1:free"

# Sample Debit Card System Description
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

# Define Specialized Agents
def create_debit_card_agents():
    """Create agents specialized for debit card system modernization"""
    
    # 1. Mainframe Specialist
    mainframe_analyst = Agent(
        role="Mainframe Payment Systems Architect",
        goal="Analyze COBOL-based debit card processing systems",
        backstory=(
            "You are a senior mainframe architect with 15+ years experience in card payment systems. "
            "You specialize in reverse-engineering COBOL applications and mapping them to modern architectures."
        ),
        llm=openrouter_model_id,
        verbose=True
    )
    
    # 2. Payments Modernizer
    payments_architect = Agent(
        role="Payments Modernization Expert",
        goal="Design cloud-native replacements for card processing components",
        backstory=(
            "Former Visa architect who specializes in PCI-compliant cloud migrations "
            "of authorization systems and fraud detection platforms."
        ),
        llm=openrouter_model_id,
        verbose=True
    )
    
    # 3. Banking Risk Analyst
    risk_analyst = Agent(
        role="Banking Migration Risk Specialist",
        goal="Identify and mitigate risks in payment system migrations",
        backstory=(
            "CTO with experience in 10+ core banking migrations, specializing in "
            "payment systems and regulatory compliance (PSD2, PCI-DSS)."
        ),
        llm=openrouter_model_id,
        verbose=True
    )
    
    return [mainframe_analyst, payments_architect, risk_analyst]

# Define Specialized Tasks
def create_debit_card_tasks(system_description: str, agents: list):
    """Create tasks for debit card system analysis"""
    
    # Task 1: Current State Analysis
    analysis_task = Task(
        description=(
            f"Analyze this debit card processing system:\n\n{system_description}\n\n"
            "Focus on:\n"
            "- Current architecture and dependencies\n"
            "- Transaction processing flows\n"
            "- Batch vs real-time operations\n"
            "- Mainframe-specific challenges"
        ),
        expected_output=(
            "Detailed report with:\n"
            "1. Current architecture diagram\n"
            "2. Key processing flows\n"
            "3. Mainframe-specific constraints\n"
            "4. Immediate pain points"
        ),
        agent=agents[0],  # Mainframe analyst
        output_file="current_state.md"
    )
    
    # Task 2: Target Architecture
    design_task = Task(
        description=(
            "Design a modern architecture for this debit card system that:\n"
            "1. Maintains existing transaction SLAs\n"
            "2. Enables real-time capabilities\n"
            "3. Reduces mainframe dependency\n"
            "4. Complies with PCI-DSS standards\n\n"
            "Base your design on the analysis from the previous task."
        ),
        expected_output=(
            "Target architecture with:\n"
            "1. Service decomposition\n"
            "2. Technology recommendations\n"
            "3. Integration approach\n"
            "4. Compliance considerations"
        ),
        agent=agents[1],  # Payments architect
        context=[analysis_task],
        output_file="target_architecture.md"
    )
    
    # Task 3: Risk Assessment
    risk_task = Task(
        description=(
            "Identify risks in migrating this debit card system to the proposed architecture.\n"
            "Consider:\n"
            "- Transaction consistency\n"
            "- Certification requirements\n"
            "- Regulatory compliance\n"
            "- Operational resilience"
        ),
        expected_output=(
            "Risk assessment with:\n"
            "1. Risk matrix (likelihood/impact)\n"
            "2. Mitigation strategies\n"
            "3. Critical success factors"
        ),
        agent=agents[2],  # Risk analyst
        context=[analysis_task, design_task],
        output_file="risk_assessment.md"
    )
    
    return [analysis_task, design_task, risk_task]

# Enhanced Streamlit UI
def run_debit_card_analyzer():
    st.set_page_config(layout="wide", page_title="Debit Card Modernizer", page_icon="💳")
    
    # Custom styling
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 25px;
            background-color: #F0F2F6;
            border-radius: 10px 10px 0px 0px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
        }
        .result-card {
            border-left: 5px solid #2E86AB;
            padding: 15px;
            background-color: #F8F9FA;
            margin-bottom: 15px;
            border-radius: 0 8px 8px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("💳 Debit Card System Modernizer")
    st.caption("For Lloyd's Bank - Prepared for Goldy Samra")
    
    # System input
    with st.expander("🔍 System Description", expanded=True):
        system_description = st.text_area(
            "Describe your debit card processing system:",
            height=250,
            value=DEFAULT_SYSTEM_DESCRIPTION,
            label_visibility="collapsed"
        )
    
    # Analysis controls
    if st.button("Analyze System", type="primary"):
        if not system_description.strip():
            st.error("Please enter a system description")
            return
            
        with st.spinner("Analyzing with specialized agents..."):
            # Create agents and tasks
            agents = create_debit_card_agents()
            tasks = create_debit_card_tasks(system_description, agents)
            
            # Create and run crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                manager_llm=openrouter_model_id,
                process=Process.sequential,
                verbose=True
            )
            
            try:
                # Simulated results for demo (replace with crew.kickoff() in production)
                results = {
                    "current_state": {
                        "architecture": {
                            "nodes": ["Authorization", "Account Balancer", "Fraud Monitor"],
                            "edges": [("Authorization", "Account Balancer"), ("Authorization", "Fraud Monitor")]
                        },
                        "findings": [
                            "Tight coupling between authorization and account balancing",
                            "Batch processing creates 4-hour latency for fraud detection",
                            "Hardcoded business rules require COBOL specialists for changes"
                        ]
                    },
                    "target_architecture": {
                        "services": [
                            {"name": "Payment Auth Service", "tech": ["Java", "Kafka"], "responsibility": "Real-time transaction approval"},
                            {"name": "Fraud Detection", "tech": ["Python", "ML"], "responsibility": "Real-time anomaly scoring"},
                            {"name": "Account Service", "tech": ["Kotlin", "PostgreSQL"], "responsibility": "Balance management"}
                        ],
                        "compliance": ["PCI-DSS Level 1", "PSD2 SCA"]
                    },
                    "risk_assessment": {
                        "high_risk": [
                            "Visa/MasterCard recertification",
                            "Transaction consistency during migration",
                            "PCI-DSS compliance in new architecture"
                        ],
                        "mitigations": [
                            "Create certification test harness early",
                            "Implement dual-write pattern during transition",
                            "Engage QSA during design phase"
                        ]
                    }
                }
                
                # Store results
                st.session_state.results = results
                st.success("✅ Analysis completed!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display results in tabs if available
    if 'results' in st.session_state:
        tab1, tab2, tab3 = st.tabs(["Current State", "Target Architecture", "Risk Assessment"])
        
        with tab1:
            st.subheader("Current System Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Architecture Diagram")
                graph = graphviz.Digraph()
                for node in st.session_state.results["current_state"]["architecture"]["nodes"]:
                    graph.node(node)
                for edge in st.session_state.results["current_state"]["architecture"]["edges"]:
                    graph.edge(edge[0], edge[1])
                st.graphviz_chart(graph)
            
            with col2:
                st.markdown("#### Key Findings")
                for finding in st.session_state.results["current_state"]["findings"]:
                    st.markdown(f"- 🔴 {finding}")
        
        with tab2:
            st.subheader("Proposed Architecture")
            
            st.markdown("#### Recommended Services")
            df_services = pd.DataFrame(st.session_state.results["target_architecture"]["services"])
            st.dataframe(df_services, hide_index=True, use_container_width=True)
            
            st.markdown("#### Compliance Requirements")
            for req in st.session_state.results["target_architecture"]["compliance"]:
                st.markdown(f"- ✅ {req}")
        
        with tab3:
            st.subheader("Migration Risk Assessment")
            
            risk_df = pd.DataFrame({
                "Risk": st.session_state.results["risk_assessment"]["high_risk"],
                "Mitigation": st.session_state.results["risk_assessment"]["mitigations"],
                "Severity": ["High", "Critical", "High"]
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

if __name__ == "__main__":
    run_debit_card_analyzer()
