import os
import streamlit as st
import time
import networkx as nx
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from openai import OpenAI
import logging
import requests
import litellm
from litellm import completion

# Configure LiteLLM
litellm.drop_params = True  # Ignore unsupported params

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="Legacy System Decomposition Analyzer", layout="wide")

# Title and description
st.title("🏦 Banking Legacy System Decomposition Analyzer")
st.markdown("""
This tool analyzes monolithic banking applications and recommends microservice decomposition strategies.
Upload your system description or use our sample banking system to generate a comprehensive migration plan.
""")

# Initialize session state variables if they don't exist
if 'result' not in st.session_state:
    st.session_state.result = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'step_results' not in st.session_state:
    st.session_state.step_results = []
if 'progress' not in st.session_state:
    st.session_state.progress = 0

def check_deepseek_api_status(api_key):
    """Check if DeepSeek API is available and credentials are valid"""
    try:
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "test"}],
            api_key=api_key,
            max_tokens=1
        )
        return True, "API is available and credentials are valid"
    except Exception as e:
        return False, f"API check failed: {str(e)}"

# Custom DeepSeek LLM class for CrewAI compatibility
class DeepSeekChatModel:
    def __init__(self, api_key, model="deepseek-chat", temperature=0.2):
        self.api_key = api_key
        self.model = f"deepseek/{model}"  # Add provider prefix
        self.temperature = temperature

    def __call__(self, prompt, stop=None):
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                api_key=self.api_key,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return f"Error: {str(e)}"

# Function to create a dependency graph visualization
def create_dependency_graph(components):
    G = nx.DiGraph()
    
    # Add nodes
    for component in components:
        G.add_node(component["name"])
    
    # Add edges based on dependencies
    for component in components:
        for dependency in component.get("dependencies", []):
            G.add_edge(component["name"], dependency)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, 
            font_size=10, font_weight='bold', arrows=True, arrowsize=20)
    
    # Save the figure to a temporary buffer
    plt.tight_layout()
    return plt

# Function to initialize and run CrewAI analysis with DeepSeek
def run_analysis(legacy_system_description, api_key, model="deepseek-chat"):
    try:

        deepseek_llm = DeepSeekChatModel(
            api_key=api_key, 
            model=model
        )

        # First verify the model is available
        available_models = requests.get(
            "https://api.deepseek.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        ).json()
        
        available_model_ids = [m["id"] for m in available_models.get("data", [])]
        if model not in available_model_ids:
            raise ValueError(
                f"Model {model} not available. Available models: {available_model_ids}\n"
                "Please select one of the available models from the sidebar."
            )

        # Reset progress
        st.session_state.progress = 0
        st.session_state.step_results = []
        st.session_state.current_step = 0
        
        # Initialize the DeepSeek LLM
        deepseek_llm = DeepSeekChatModel(api_key=api_key, model=model)
        
        # Define the agents
        code_analyzer = Agent(
            role="Code Analyzer",
            goal="Analyze legacy codebase to identify component dependencies and potential service boundaries",
            backstory="""You are an expert in static code analysis with deep experience in banking systems.
            You excel at understanding complex codebases and identifying natural break points.""",
            llm=deepseek_llm,
            verbose=True
        )
        
        data_flow_mapper = Agent(
            role="Data Flow Mapper",
            goal="Map data flows between components and identify data ownership patterns",
            backstory="""You specialize in database analysis and data flow tracking in large systems.
            You can identify which components own what data and how data moves through the system.""",
            llm=deepseek_llm,
            verbose=True
        )
        
        domain_expert = Agent(
            role="Banking Domain Expert",
            goal="Map technical components to banking business capabilities",
            backstory="""You have 15 years of experience in banking systems architecture.
            You understand how technical components relate to business functions like account management,
            transactions, loans, and compliance.""",
            llm=deepseek_llm,
            verbose=True
        )
        
        architecture_strategist = Agent(
            role="Microservice Architecture Strategist",
            goal="Define optimal microservice boundaries based on technical and business analysis",
            backstory="""You are a leading expert in microservice architecture design.
            You excel at finding the right service boundaries that balance technical constraints
            with business needs, especially in highly regulated environments like banking.""",
            llm=deepseek_llm,
            verbose=True
        )
        
        risk_assessor = Agent(
            role="Migration Risk Assessor",
            goal="Evaluate risks associated with the proposed decomposition strategy",
            backstory="""You are specialized in risk management for large-scale system migrations.
            You can identify potential failure points and suggest mitigation strategies for
            high-stakes systems like those in banking.""",
            llm=deepseek_llm,
            verbose=True
        )
        
        # Define task functions with progress updates
        def execute_task(task):
            try:
                # Get the agent assigned to this task
                agent = next(agent for agent in legacy_decomposition_crew.agents if agent.role == task.agent.role)
                
                # Execute the task using the agent
                task_result = agent.execute_task(task)
                
                st.session_state.current_step += 1
                st.session_state.progress = st.session_state.current_step / 6  # 6 total tasks
                st.session_state.step_results.append({
                    "agent": task.agent.role,
                    "result": task_result
                })
                return task_result
            except Exception as e:
                logger.error(f"Error executing task: {e}")
                return f"Error executing task: {str(e)}"
        
        # Define the tasks
        code_analysis_task = Task(
            description=f"""
            Analyze the legacy system codebase described below and identify:
            1. Major components and their responsibilities
            2. Dependencies between components (who calls whom)
            3. Potential natural service boundaries based on code structure
            4. Technical debt areas that need special attention
            
            Legacy System:
            {legacy_system_description}
            
            Provide a detailed analysis with component dependency graph in text format.
            List each component with its name, responsibility, and dependencies in a structured format.
            """,
            expected_output="A structured markdown report listing each component with its name, responsibilities, and dependencies.",
            agent=code_analyzer
        )
        
        data_flow_task = Task(
            description=f"""
            Analyze the data flow patterns in the legacy system described below:
            1. Identify which components own what data
            2. Map how data flows between components
            3. Identify transactional boundaries
            4. Detect potential data coupling issues that might complicate decomposition
            
            Legacy System:
            {legacy_system_description}
            
            Use the code analysis provided by the Code Analyzer to inform your analysis.
            Provide a detailed data flow map in text format.
            """,
            expected_output="A detailed explanation of data ownership and flow in structured text format.",
            agent=data_flow_mapper,
            context=[code_analysis_task]
        )
        
        domain_mapping_task = Task(
            description=f"""
            Map the technical components of the legacy system to banking business capabilities:
            1. Identify which business capabilities each component supports
            2. Assess the business criticality of each component
            3. Identify cross-cutting business processes that span multiple components
            
            Legacy System:
            {legacy_system_description}
            
            Use the analyses provided by the Code Analyzer and Data Flow Mapper to inform your mapping.
            Provide a detailed business capability map in text format.
            """,
            expected_output="A business capability map linking technical components to banking functions in bullet-point format.",
            agent=domain_expert,
            context=[code_analysis_task, data_flow_task]
        )
        
        microservice_design_task = Task(
            description="""
            Based on the code analysis, data flow mapping, and domain mapping:
            1. Propose optimal microservice boundaries
            2. Justify each boundary decision
            3. Recommend API patterns for inter-service communication
            4. Suggest a target state architecture diagram
            
            Provide a detailed microservice architecture proposal with justifications.
            For each proposed microservice, list its name, responsibilities, data ownership, and APIs.
            """,
            expected_output="A proposal listing each microservice, responsibilities, APIs, and architecture recommendations.",
            agent=architecture_strategist,           
            context=[code_analysis_task, data_flow_task, domain_mapping_task]
        )
        
        risk_assessment_task = Task(
            description="""
            Evaluate the risks associated with the proposed microservice decomposition:
            1. Identify high-risk migration areas
            2. Assess potential performance impacts
            3. Evaluate data consistency challenges
            4. Consider regulatory compliance implications specific to banking
            
            Provide a comprehensive risk assessment with risk levels (High/Medium/Low) for each area.
            Include specific mitigation strategies for each identified risk.
            """,
            expected_output="A table or list of migration risks with risk levels and mitigation strategies.",
            agent=risk_assessor,
            context=[microservice_design_task]
        )
        
        migration_roadmap_task = Task(
            description="""
            Based on the microservice architecture proposal and risk assessment:
            1. Create a phased migration roadmap
            2. Prioritize which services should be extracted first
            3. Suggest implementation approaches for each phase
            4. Recommend testing and validation strategies specific to banking systems
            
            Provide a detailed migration plan with phases, timeline estimates, and key milestones.
            Format your roadmap as a clear phase-by-phase plan with specific recommendations.
            """,
            expected_output="A phased migration roadmap with timeline, milestones, and testing strategies.",
            agent=architecture_strategist,
            context=[microservice_design_task, risk_assessment_task]
        )
        
        # Create the crew
        legacy_decomposition_crew = Crew(
            agents=[code_analyzer, data_flow_mapper, domain_expert, architecture_strategist, risk_assessor],
            tasks=[],  # We'll execute tasks manually for progress tracking
            verbose=1
        )
        
        # Execute tasks sequentially with progress updates
        tasks = [code_analysis_task, data_flow_task, domain_mapping_task, 
                microservice_design_task, risk_assessment_task, migration_roadmap_task]
        
        all_results = []
        for task in tasks:
            result = execute_task(task)
            all_results.append(result)
        
        combined_result = "\n\n".join(all_results)
        st.session_state.result = combined_result
        st.session_state.analysis_complete = True
        
        return combined_result
    
    except Exception as e:
        logger.error(f"Error in run_analysis: {e}")
        st.error(f"An error occurred during analysis: {str(e)}")
        return f"Error: {str(e)}"

# Sidebar for system configuration
st.sidebar.header("Configuration")

# API Key input
api_key = st.sidebar.text_input("DeepSeek API Key", type="password")

# Model selection
model = st.sidebar.selectbox(
    "Select Model",
    ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
    index=0
)

# Example system or custom input
system_option = st.sidebar.radio(
    "System Description",
    ["Use Sample Banking System", "Custom System Description"]
)

if system_option == "Use Sample Banking System":
    legacy_system_description = """
    The legacy system is a 15-year-old monolithic Java application handling retail banking operations for Lloyds Bank.
    Key components include:
    1. CustomerManagement - stores and manages customer profiles, with ~500K LOC
    2. AccountServices - handles checking, savings accounts with ~800K LOC
    3. LoanProcessing - manages loan applications and servicing with ~600K LOC
    4. PaymentSystem - processes various payment types with ~400K LOC
    5. ReportingModule - generates regulatory and business reports with ~300K LOC
    6. NotificationService - handles customer communications via email, SMS with ~200K LOC
    7. SecurityModule - manages authentication, authorization with ~350K LOC
    8. ComplianceEngine - enforces regulatory rules and audit trails with ~450K LOC

    These components share a single Oracle database with 200+ tables.
    Inter-component communication happens via direct method calls.
    The system processes approximately 2M transactions daily with peaks of 200 TPS.
    The system has high availability requirements (99.99%) and is deployed in active-passive configuration.
    Regulatory compliance requirements include PSD2, GDPR, and UK banking regulations.
    """
    st.text_area("Sample System Description", legacy_system_description, height=300, disabled=True)
else:
    legacy_system_description = st.text_area(
        "Enter your legacy system description",
        height=300,
        placeholder="Describe your legacy banking system components, database structure, integration points, etc."
    )

# Run analysis button
if st.sidebar.button("Run Analysis"):
    if not api_key:
        st.sidebar.error("Please enter a DeepSeek API key")
    elif not legacy_system_description:
        st.sidebar.error("Please enter a system description")
    else:
        # Check API status before proceeding
        with st.spinner("Checking DeepSeek API status..."):
            api_ok, api_message = check_deepseek_api_status(api_key)
            
        if api_ok:
            with st.spinner("Running analysis... This may take several minutes"):
                try:
                    run_analysis(legacy_system_description, api_key, model)
                except ValueError as e:
                    st.error(str(e))
                    logger.error(str(e))
        else:
            st.error(f"DeepSeek API unavailable: {api_message}")
            logger.error(f"DeepSeek API check failed: {api_message}")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Analysis Progress", "Results", "Visualizations", "Migration Plan"])

# Progress bar in tab 1
with tab1:
    if 'progress' in st.session_state:
        st.subheader("Analysis Progress")
        progress_bar = st.progress(st.session_state.progress)
        
        # Show current step
        steps = ["Code Analysis", "Data Flow Mapping", "Domain Mapping", 
                 "Microservice Design", "Risk Assessment", "Migration Roadmap"]
        current_step = min(st.session_state.current_step, len(steps) - 1)
        if current_step >= 0:
            st.markdown(f"**Current Step:** {steps[current_step]}")
        
        # Show step results as they come in
        for i, step_result in enumerate(st.session_state.step_results):
            with st.expander(f"{steps[i]} - {step_result['agent']}", expanded=i == current_step):
                st.markdown(step_result["result"])

# Display full results in tab 2
with tab2:
    if st.session_state.analysis_complete:
        st.subheader("Complete Analysis Results")
        st.markdown(st.session_state.result)
    else:
        st.info("Run the analysis to see results here")

# Visualizations in tab 3
with tab3:
    if st.session_state.analysis_complete:
        st.subheader("System Visualizations")
        
        # Create tabs for different visualizations
        vis_tab1, vis_tab2 = st.tabs(["Component Dependencies", "Proposed Microservices"])
        
        with vis_tab1:
            st.markdown("### Legacy System Component Dependencies")
            sample_components = [
                {"name": "CustomerManagement", "dependencies": ["SecurityModule"]},
                {"name": "AccountServices", "dependencies": ["CustomerManagement", "NotificationService"]},
                {"name": "LoanProcessing", "dependencies": ["CustomerManagement", "ComplianceEngine", "NotificationService"]},
                {"name": "PaymentSystem", "dependencies": ["AccountServices", "SecurityModule"]},
                {"name": "ReportingModule", "dependencies": ["AccountServices", "LoanProcessing", "PaymentSystem"]},
                {"name": "NotificationService", "dependencies": ["SecurityModule"]},
                {"name": "SecurityModule", "dependencies": []},
                {"name": "ComplianceEngine", "dependencies": ["SecurityModule", "ReportingModule"]}
            ]
            
            dependency_graph = create_dependency_graph(sample_components)
            st.pyplot(dependency_graph)
        
        with vis_tab2:
            st.markdown("### Proposed Microservice Architecture")
            st.image("https://via.placeholder.com/800x600.png?text=Proposed+Microservice+Architecture", 
                    caption="Example Microservice Architecture")
            st.markdown("""
            **Note:** In the actual implementation, this would be a dynamic visualization
            generated from the architecture strategist's recommendations.
            """)
    else:
        st.info("Run the analysis to see visualizations here")

# Migration plan in tab 4
with tab4:
    if st.session_state.analysis_complete:
        st.subheader("Migration Roadmap")
        
        st.markdown("""
        ### Phase 1: Foundation (Months 1-3)
        - Establish API gateway
        - Set up CI/CD pipeline
        - Create monitoring infrastructure
        - Extract SecurityModule as first microservice
        
        ### Phase 2: Customer Domain (Months 4-6)
        - Extract CustomerManagement service
        - Extract NotificationService
        - Implement BFF (Backend for Frontend) pattern
        
        ### Phase 3: Account Services (Months 7-10)
        - Extract AccountServices into multiple microservices:
          - Account Management
          - Transaction Processing
          - Interest Calculation
        - Implement event-driven communication pattern
        
        ### Phase 4: Loans & Payments (Months 11-15)
        - Extract LoanProcessing services
        - Extract PaymentSystem services
        - Implement saga pattern for distributed transactions
        
        ### Phase 5: Reporting & Compliance (Months 16-18)
        - Extract ReportingModule services
        - Extract ComplianceEngine services
        - Implement data lake for analytics
        
        ### Phase 6: Legacy Retirement (Months 19-24)  
        - Complete testing of all microservices
        - Gradual traffic migration
        - Decommission monolith
        """)
        
        # Timeline visualization
        st.markdown("### Migration Timeline")
        timeline_data = {
            "Phase 1: Foundation": 3,
            "Phase 2: Customer Domain": 3,
            "Phase 3: Account Services": 4,
            "Phase 4: Loans & Payments": 5,
            "Phase 5: Reporting & Compliance": 3,
            "Phase 6: Legacy Retirement": 6
        }
        
        # Create a horizontal bar chart for the timeline
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(timeline_data))
        phases = list(timeline_data.keys())
        durations = list(timeline_data.values())
        
        ax.barh(y_pos, durations, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(phases)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Months')
        ax.set_title('Migration Timeline')
        
        # Add duration labels to the end of each bar
        for i, v in enumerate(durations):
            ax.text(v + 0.1, i, str(v) + " months", va='center')
        
        st.pyplot(fig)
    else:
        st.info("Run the analysis to see the migration plan here")

# Footer with DeepSeek attribution
st.markdown("---")
st.markdown("""
**Note**: This Legacy System Decomposition Analyzer uses DeepSeek AI for analysis.
In a production environment, this would connect to actual code repositories and database schemas.
""")

# Error handling information
st.sidebar.markdown("---")
st.sidebar.markdown("### Troubleshooting")
st.sidebar.markdown("""
If you encounter errors:
1. Verify your DeepSeek API key is correct
2. Check your internet connection
3. Try reducing the complexity of your system description
""")
