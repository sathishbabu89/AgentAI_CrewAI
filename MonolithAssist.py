import os
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew
import logging
import requests
import litellm
from litellm import completion
import zipfile
import tempfile
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import traceback


# Configure LiteLLM
litellm.drop_params = True  # Ignore unsupported params

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "deepseek-coder"
AVAILABLE_MODELS = ["deepseek-coder", "deepseek-chat", "deepseek-reasoner"]
ARCHITECTURE_TEMPLATES = {
    "Spring Boot": "spring_boot_template",
    "Node.js": "nodejs_template",
    "Python": "python_template"
}

# Set page configuration
st.set_page_config(
    page_title="Monolith to Microservices Converter",
    layout="wide",
    page_icon="🔄"
)

class FileProcessor:
    """Handles file uploads and processing"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return None
    
    @staticmethod
    def extract_zip(zip_path, extract_to):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            logger.error(f"Error extracting zip: {e}")
            return False
    
    @staticmethod
    def analyze_code_structure(code_dir):
        """Basic code structure analysis (placeholder for actual analysis)"""
        structure = {
            "languages": [],
            "files": 0,
            "directories": 0,
            "entry_points": []
        }
        
        for root, dirs, files in os.walk(code_dir):
            structure["directories"] += len(dirs)
            for file in files:
                structure["files"] += 1
                if file.endswith(('.java', '.py', '.js')):
                    if "main" in file.lower() or "app" in file.lower():
                        structure["entry_points"].append(os.path.join(root, file))
        
        return structure

class SessionStateManager:
    """Manages session state variables"""
    
    @staticmethod
    def initialize():
        if 'result' not in st.session_state:
            st.session_state.result = None
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'progress' not in st.session_state:
            st.session_state.progress = 0
        if 'components' not in st.session_state:
            st.session_state.components = []
        if 'last_api_check' not in st.session_state:
            st.session_state.last_api_check = 0
        if 'code_structure' not in st.session_state:
            st.session_state.code_structure = {}
        if 'extracted_code_path' not in st.session_state:
            st.session_state.extracted_code_path = ""
        if 'microservices_design' not in st.session_state:
            st.session_state.microservices_design = {}

class DeepSeekAPI:
    """Handles all DeepSeek API interactions"""
    
    @staticmethod
    def check_api_status(api_key: str) -> Tuple[bool, str]:
        """Check if DeepSeek API is available and credentials are valid"""
        try:
            response = completion(
                model="deepseek/deepseek-coder",
                messages=[{"role": "user", "content": "test"}],
                api_key=api_key,
                max_tokens=1
            )
            st.session_state.last_api_check = time.time()
            return True, "API is available and credentials are valid"
        except Exception as e:
            return False, f"API check failed: {str(e)}"
    
    @staticmethod
    def get_available_models(api_key: str) -> List[str]:
        """Get list of available models from DeepSeek API"""
        try:
            response = requests.get(
                "https://api.deepseek.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            return [m["id"] for m in response.json().get("data", [])]
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []

class DeepSeekChatModel:
    """Custom DeepSeek LLM class for CrewAI compatibility"""
    
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.2):
        self.api_key = api_key
        self.model = f"deepseek/{model}"  # Add provider prefix
        self.temperature = temperature
    
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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

class MicroserviceDesigner:
    """Coordinates the microservice conversion process"""
    
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.llm = DeepSeekChatModel(api_key=api_key, model=model)
    
    def create_agents(self):
        """Create all the specialist agents for the conversion process"""
        
        # Code Understanding Agent
        self.code_analyzer = Agent(
            role="Senior Code Architect",
            goal="Analyze monolithic codebase to extract structure, dependencies, and domain models",
            backstory="""You are an expert in static code analysis with 20+ years experience in enterprise systems.
            You excel at understanding complex codebases and identifying domain boundaries.""",
            llm=self.llm,
            verbose=True
        )
        
        # Domain Decomposition Agent
        self.domain_designer = Agent(
            role="Domain-Driven Design Expert",
            goal="Identify bounded contexts and define microservice boundaries",
            backstory="""You are a DDD specialist with deep knowledge of strategic design patterns.
            You decompose monoliths into cohesive microservices using subdomain analysis.""",
            llm=self.llm,
            verbose=True
        )
        
        # Service Generator Agent
        self.service_generator = Agent(
            role="Microservice Developer",
            goal="Generate Spring Boot microservices with clean architecture",
            backstory="""You are a senior Java developer specializing in microservice architecture.
            You implement services using Spring Boot, Hexagonal Architecture, and cloud-native patterns.""",
            llm=self.llm,
            verbose=True
        )
        
        # Testing Specialist Agent
        self.testing_specialist = Agent(
            role="Quality Assurance Architect",
            goal="Create comprehensive test suites for microservices",
            backstory="""You are a testing expert with experience in unit, integration, and contract testing.
            You ensure services are reliable and meet quality standards.""",
            llm=self.llm,
            verbose=True
        )
        
        # DevOps Engineer Agent
        self.devops_engineer = Agent(
            role="DevOps Specialist",
            goal="Configure CI/CD pipelines and observability tools",
            backstory="""You are a cloud infrastructure expert specializing in Kubernetes and observability.
            You automate deployment and monitoring of microservices.""",
            llm=self.llm,
            verbose=True
        )
    
    # Modify the analyze_monolith method in the MicroserviceDesigner class:
    def analyze_monolith(self, code_dir: str):
        """Phase 1: Understand the monolith"""
        analysis_task = Task(
            description=f"""
            Analyze the monolithic codebase located at: {code_dir}
            
            Perform comprehensive analysis including:
            1. Code structure and architecture
            2. Class relationships and dependencies
            3. Database schema and usage patterns
            4. Entry points and service boundaries
            
            Identify:
            - Potential domain models
            - Existing service boundaries
            - Data access patterns
            - Transaction boundaries
            
            Produce:
            - Class diagrams (Mermaid format)
            - Call graphs
            - Database schema maps
            - Code quality metrics
            
            IMPORTANT: Your response MUST be valid JSON that follows this exact structure:
            {{
                "code_structure": {{
                    "languages": [],
                    "modules": [],
                    "entry_points": []
                }},
                "domain_analysis": {{
                    "potential_domains": [],
                    "bounded_contexts": [],
                    "transaction_boundaries": []
                }},
                "visualizations": {{
                    "class_diagram": "mermaid_code",
                    "call_graph": "mermaid_code",
                    "db_schema": "mermaid_code"
                }},
                "metrics": {{
                    "cyclomatic_complexity": {{
                        "average": 0,
                        "max": 0,
                        "problem_files": []
                    }},
                    "coupling": {{
                        "afferent": 0,
                        "efferent": 0
                    }},
                    "cohesion": {{
                        "LCOM4": 0
                    }}
                }},
                "recommendations": []
            }}
            
            If you cannot analyze any part, provide empty arrays/objects but maintain the structure.
            """,
            expected_output="Comprehensive analysis of the monolith in JSON format",
            agent=self.code_analyzer
        )
        
        try:
            raw_result = self.code_analyzer.execute_task(analysis_task)
            
            # Clean the response to extract just the JSON portion
            json_start = raw_result.find('{')
            json_end = raw_result.rfind('}') + 1
            json_str = raw_result[json_start:json_end]
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Validate the structure
            if not all(key in result for key in ['code_structure', 'domain_analysis', 'visualizations']):
                raise ValueError("Analysis result missing required sections")
                
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nResponse was: {raw_result}")
            st.error("Failed to parse analysis results. The AI response was malformed.")
            return json.dumps({
                "error": "Analysis failed",
                "details": str(e),
                "raw_response": raw_result[:500] + "..." if len(raw_result) > 500 else raw_result
            })
        except Exception as e:
            logger.error(f"Analysis failed: {traceback.format_exc()}")
            st.error(f"Analysis failed: {str(e)}")
            return json.dumps({"error": str(e)})
    
    # Also modify the design_microservices method similarly:
    def design_microservices(self, analysis_result: str):
        """Phase 2: Define microservice boundaries"""
        design_task = Task(
            description=f"""
            Based on this monolith analysis:
            {analysis_result}
            
            Design a microservice architecture using Domain-Driven Design principles:
            
            1. Decompose into bounded contexts
            2. Define service boundaries
            3. Apply database-per-service pattern
            4. Identify shared kernel areas
            5. Define integration patterns
            
            Consider:
            - Business capabilities
            - Data ownership boundaries
            - Transaction requirements
            - Team organization
            
            Output MUST be valid JSON with this exact structure:
            {{
                "microservices": [
                    {{
                        "name": "service_name",
                        "responsibilities": [],
                        "bounded_context": "",
                        "database": {{
                            "type": "",
                            "schema": ""
                        }},
                        "dependencies": [],
                        "integration_type": "sync/async"
                    }}
                ],
                "integration": {{
                    "api_gateway": true/false,
                    "event_bus": "Kafka/RabbitMQ/none",
                    "saga_patterns": []
                }},
                "architecture_diagram": "mermaid_code",
                "deployment_recommendations": []
            }}
            
            If unsure about any field, provide reasonable defaults but maintain the structure.
            """,
            expected_output="Microservice design in JSON format",
            agent=self.domain_designer
        )
        
        try:
            raw_result = self.domain_designer.execute_task(design_task)
            
            # Clean the response to extract just the JSON portion
            json_start = raw_result.find('{')
            json_end = raw_result.rfind('}') + 1
            json_str = raw_result[json_start:json_end]
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Validate the structure
            if "microservices" not in result:
                raise ValueError("Design result missing microservices section")
                
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nResponse was: {raw_result}")
            st.error("Failed to parse design results. The AI response was malformed.")
            return json.dumps({
                "error": "Design failed",
                "details": str(e),
                "raw_response": raw_result[:500] + "..." if len(raw_result) > 500 else raw_result
            })
        except Exception as e:
            logger.error(f"Design failed: {traceback.format_exc()}")
            st.error(f"Design failed: {str(e)}")
            return json.dumps({"error": str(e)})

    def _create_zip_from_output(self, generated_output: str) -> str:
        """Convert the generated output into a downloadable zip file"""
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "microservices.zip")
        
        current_path = None
        current_content = []
        
        # Parse the generated output to create files
        for line in generated_output.split('\n'):
            if line.startswith('microservices/'):
                # This is a new file path
                if current_path and current_content:
                    self._write_file(temp_dir, current_path, current_content)
                
                current_path = line.strip()
                current_content = []
            else:
                if current_path:  # Only collect content if we have a path
                    current_content.append(line)
        
        # Write the last file
        if current_path and current_content:
            self._write_file(temp_dir, current_path, current_content)
        
        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file != "microservices.zip":  # Don't zip the zip file
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
        
        return zip_path

    def _write_file(self, base_dir: str, file_path: str, content_lines: list):
        """Write a single file to the temporary directory"""
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))   
    
    def generate_services(self, design_spec: str):
        """Phase 3: Generate microservice code and prepare for download"""
        generation_task = Task(
            description=f"""
            Based on this microservice design:
            {design_spec}
            
            Generate COMPLETE, IMPLEMENTED Spring Boot microservices with:
            
            1. For EACH service include ALL these files IMPLEMENTED:
               - Main application class with proper annotations
               - FULLY IMPLEMENTED domain models with all fields
               - COMPLETE ports/interfaces
               - FULLY IMPLEMENTED application services
               - COMPLETE infrastructure adapters
               - ALL REST controllers with proper validation
               - COMPLETE DTOs for all API requests/responses
               - FULL exception handling
               - COMPLETE database entities and repositories
               - ALL configuration classes
            
            2. For EACH service include:
               - COMPLETE application.yml with:
                 * Database configuration
                 * Server configuration
                 * Security configuration
                 * External service URLs
                 * Monitoring configuration
               - COMPLETE Dockerfile
               - COMPLETE Kubernetes deployment files
               - COMPLETE Helm charts
            
            3. For EACH service include:
               - COMPLETE Swagger/OpenAPI documentation
               - Health checks
               - Metrics endpoints
               - Distributed tracing
            
            4. Structure MUST be production-ready with:
               - PROPER package organization
               - COMPLETE pom.xml with all dependencies
               - ALL necessary annotations
               - COMPLETE method implementations
            
            IMPORTANT: 
            - DO NOT leave any "// TODO" comments
            - IMPLEMENT ALL methods
            - Include ALL imports
            - Use Lombok where appropriate
            - Include PROPER error handling
            - Include LOGGING
            
            Structure the output as a complete project directory with:
            - Each service in its own folder
            - README.md for each service
            - Complete build configuration
            
            Provide the file structure and all code files.
            """,
            expected_output="Complete microservice implementation with virtual file structure",
            agent=self.service_generator
        )
        
        try:
            generated_output = self.service_generator.execute_task(generation_task)
            zip_path = self._create_zip_from_output(generated_output)
            return generated_output, zip_path
        except Exception as e:
            logger.error(f"Service generation failed: {traceback.format_exc()}")
            st.error(f"Service generation failed: {str(e)}")
            return None, None
    
    def generate_tests(self, service_code: str):
        """Phase 4: Generate test suites"""
        test_task = Task(
            description=f"""
            For this microservice implementation:
            {service_code}
            
            Create comprehensive test suites including:
            
            1. Unit tests (JUnit + Mockito):
               - Domain logic
               - Service layer
               - Controller layer
               
            2. Integration tests:
               - Database integration
               - API contracts
               - External service mocks
               
            3. Consumer-driven contracts:
               - Spring Cloud Contract definitions
               
            4. Functional tests (RestAssured):
               - API endpoint validation
               
            5. Manual test scenarios (Gherkin):
               - Given/When/Then format
               - Cover all business requirements
               
            Provide the complete test implementation.
            """,
            expected_output="Comprehensive test suites for all microservices",
            agent=self.testing_specialist
        )
        
        return self.testing_specialist.execute_task(test_task)
    
    def setup_devops(self, services_and_tests: str):
        """Phase 5: Configure CI/CD and observability"""
        devops_task = Task(
            description=f"""
            For these microservices and tests:
            {services_and_tests}
            
            Configure complete DevOps infrastructure:
            
            1. CI/CD Pipeline (GitHub Actions):
               - Build and test
               - Containerization
               - Deployment to Kubernetes
               
            2. Observability:
               - Prometheus metrics
               - Grafana dashboards
               - OpenTelemetry tracing
               
            3. Monitoring:
               - Health checks
               - Alerting rules
               
            4. Infrastructure as Code:
               - Terraform scripts
               - Kubernetes manifests
               
            Provide all configuration files and setup instructions.
            """,
            expected_output="Complete DevOps configuration",
            agent=self.devops_engineer
        )
        
        return self.devops_engineer.execute_task(devops_task)

class VisualizationEngine:
    """Handles visualization of analysis results"""
    
    @staticmethod
    def create_dependency_graph(components: List[Dict]) -> plt.Figure:
        """Create a dependency graph visualization"""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for component in components:
            G.add_node(
                component["name"],
                type=component.get("type", "service"),
                complexity=component.get("complexity", "medium")
            )
        
        # Add edges
        for component in components:
            for dependency in component.get("dependencies", []):
                G.add_edge(component["name"], dependency)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=0.6)
        
        # Node colors based on type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "service")
            if node_type == "database":
                node_colors.append("lightcoral")
            elif node_type == "gateway":
                node_colors.append("gold")
            else:
                node_colors.append("skyblue")
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=2500,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=1.5
        )
        
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold'
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Service', markerfacecolor='skyblue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Database', markerfacecolor='lightcoral', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Gateway', markerfacecolor='gold', markersize=10)
        ]
        
        plt.legend(
            handles=legend_elements,
            loc='upper right',
            title="Component Types"
        )
        
        plt.title("Microservice Dependency Architecture", pad=20)
        plt.tight_layout()
        return plt
    
    @staticmethod
    def display_mermaid_diagram(mermaid_code: str):
        """Display a Mermaid diagram in Streamlit"""
        st.markdown(f"""
        ```mermaid
        {mermaid_code}
        ```
        """, unsafe_allow_html=True)

class UIComponents:
    """Handles the user interface components"""
    
    @staticmethod
    def setup_sidebar():
        """Configure the sidebar elements"""
        st.sidebar.header("Configuration")
        
        api_key = st.sidebar.text_input(
            "DeepSeek API Key",
            type="password",
            help="Enter your DeepSeek API key"
        )
        
        model = st.sidebar.selectbox(
            "Select Model",
            AVAILABLE_MODELS,
            index=0,
            help="Choose the DeepSeek model to use for analysis"
        )
        
        target_architecture = st.sidebar.selectbox(
            "Target Architecture",
            list(ARCHITECTURE_TEMPLATES.keys()),
            index=0,
            help="Choose the target architecture for microservices"
        )
        
        return api_key, model, target_architecture
    
    @staticmethod
    def show_analysis_progress(phase: str):
        """Display the analysis progress bar"""
        phases = {
            "upload": ("Uploading code...", 0.1),
            "analyzing": ("Analyzing monolith...", 0.3),
            "designing": ("Designing microservices...", 0.5),
            "generating": ("Generating code...", 0.7),
            "testing": ("Creating tests...", 0.85),
            "devops": ("Configuring DevOps...", 1.0)
        }
        
        if phase in phases:
            label, progress = phases[phase]
            st.session_state.progress = progress
            st.progress(progress)
            st.caption(label)
    
    @staticmethod
    def show_results_tabs():
        """Display the results in tabs"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "Analysis Report", 
            "Architecture Design", 
            "Generated Code",
            "DevOps Setup"
        ])
        
        with tab1:
            if st.session_state.analysis_complete:
                st.subheader("Monolith Analysis Report")
                if 'code_structure' in st.session_state:
                    st.json(st.session_state.code_structure)
                
                if 'visualizations' in st.session_state.code_structure:
                    st.subheader("Class Diagram")
                    VisualizationEngine.display_mermaid_diagram(
                        st.session_state.code_structure['visualizations']['class_diagram']
                    )
            else:
                st.info("Upload and analyze a monolith to view results.")
        
        with tab2:
            if st.session_state.analysis_complete and 'microservices_design' in st.session_state:
                st.subheader("Microservice Architecture Design")
                st.json(st.session_state.microservices_design)
                
                if 'architecture_diagram' in st.session_state.microservices_design:
                    st.subheader("Service Architecture Diagram")
                    VisualizationEngine.display_mermaid_diagram(
                        st.session_state.microservices_design['architecture_diagram']
                    )
                
                if 'microservices' in st.session_state.microservices_design:
                    st.subheader("Service Dependencies")
                    graph = VisualizationEngine.create_dependency_graph(
                        st.session_state.microservices_design['microservices']
                    )
                    st.pyplot(graph)
            else:
                st.info("Complete the analysis to view architecture design.")
        
        with tab3:
            if st.session_state.analysis_complete and 'generated_code' in st.session_state:
                st.subheader("Generated Microservice Code")
                st.code(st.session_state.generated_code, language='java')
            else:
                st.info("Complete the analysis to view generated code.")
        
        with tab4:
            if st.session_state.analysis_complete and 'devops_config' in st.session_state:
                st.subheader("DevOps Configuration")
                st.code(st.session_state.devops_config, language='yaml')
            else:
                st.info("Complete the analysis to view DevOps setup.")

# Initialize session state
SessionStateManager.initialize()

# Setup sidebar
api_key, model, target_architecture = UIComponents.setup_sidebar()

# Main file uploader
st.header("Upload Monolithic Codebase")
uploaded_file = st.file_uploader(
    "Choose a ZIP file containing your monolithic codebase",
    type="zip"
)

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("Processing uploaded file..."):
        # Save the uploaded file
        zip_path = FileProcessor.save_uploaded_file(uploaded_file)
        
        # Create temp directory for extraction
        extract_dir = tempfile.mkdtemp()
        st.session_state.extracted_code_path = extract_dir
        
        # Extract the zip file
        if FileProcessor.extract_zip(zip_path, extract_dir):
            st.success("File uploaded and extracted successfully!")
            
            # Analyze basic code structure
            st.session_state.code_structure = FileProcessor.analyze_code_structure(extract_dir)
            
            # Show basic info
            st.subheader("Code Structure Overview")
            col1, col2 = st.columns(2)
            col1.metric("Files", st.session_state.code_structure["files"])
            col2.metric("Directories", st.session_state.code_structure["directories"])
            
            if st.session_state.code_structure["entry_points"]:
                st.write("Main entry points found:")
                for entry in st.session_state.code_structure["entry_points"]:
                    st.code(entry)

# Run analysis button
if st.button("Convert to Microservices") and uploaded_file is not None:
    if not api_key:
        st.error("Please enter a DeepSeek API key")
    else:
        # Check API status if not checked recently
        if time.time() - st.session_state.last_api_check > 300:  # 5 minute cache
            with st.spinner("Checking DeepSeek API status..."):
                api_ok, api_message = DeepSeekAPI.check_api_status(api_key)
        else:
            api_ok = True
        
        if api_ok:
            try:
                # Initialize the designer
                designer = MicroserviceDesigner(api_key, model)
                designer.create_agents()
                
                # Phase 1: Analyze monolith
                UIComponents.show_analysis_progress("analyzing")
                analysis_result = designer.analyze_monolith(st.session_state.extracted_code_path)
                st.session_state.code_structure = json.loads(analysis_result)
                
                # Phase 2: Design microservices
                UIComponents.show_analysis_progress("designing")
                design_spec = designer.design_microservices(analysis_result)
                st.session_state.microservices_design = json.loads(design_spec)
                
                # Phase 3: Generate services
                UIComponents.show_analysis_progress("generating")
                generated_code, zip_path = designer.generate_services(design_spec)
                st.session_state.generated_code = generated_code
                st.session_state.zip_path = zip_path
                
                # Phase 4: Generate tests
                UIComponents.show_analysis_progress("testing")
                test_suite = designer.generate_tests(generated_code)
                st.session_state.test_suite = test_suite
                
                # Phase 5: DevOps setup
                UIComponents.show_analysis_progress("devops")
                devops_config = designer.setup_devops(generated_code + "\n" + test_suite)
                st.session_state.devops_config = devops_config
                
                st.session_state.analysis_complete = True
                st.balloons()
                st.success("Microservice conversion complete!")
                
                # Display download button if zip file was created
                if 'zip_path' in st.session_state and st.session_state.zip_path:
                    with open(st.session_state.zip_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Microservices as ZIP",
                            data=f,
                            file_name="microservices.zip",
                            mime="application/zip",
                            help="Download all generated microservices as a zip file",
                            key="download_zip"
                        )
                
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")
                logger.error(f"Conversion error: {str(e)}")
        else:
            st.error(f"DeepSeek API unavailable: {api_message}")

# Display progress if analysis is running
if st.session_state.progress > 0 and st.session_state.progress < 1:
    UIComponents.show_analysis_progress("analyzing")

# Display results
UIComponents.show_results_tabs()

# Cleanup temp files on app rerun
if 'extracted_code_path' in st.session_state and st.session_state.extracted_code_path:
    try:
        shutil.rmtree(st.session_state.extracted_code_path)
    except Exception as e:
        logger.warning(f"Could not cleanup temp dir: {e}")

# Cleanup zip file if it exists
if 'zip_path' in st.session_state and st.session_state.zip_path:
    try:
        os.remove(st.session_state.zip_path)
        temp_dir = os.path.dirname(st.session_state.zip_path)
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Could not cleanup zip file: {e}")

# Footer
st.markdown("---")
st.markdown("""
**Note**: This tool uses AI to analyze and convert monolithic applications to microservices.
Always review generated code before deployment in production environments.
""")
