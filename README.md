# Codex7-Agent

Codex7-Agent is an advanced AI agent framework designed for software engineering tasks, featuring a dynamic agent loop with intelligent tool orchestration and SubAgent mechanisms for complex task handling.

## Table of Contents
- [Architecture](#architecture)
- [Key Innovations](#key-innovations)
- [Usage Flow](#usage-flow)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Tools](#tools)
- [Configuration](#configuration)

## Architecture

The Codex7-Agent follows a modular architecture that separates concerns while maintaining tight integration between components:

```
codex7-agent/
├── src/
│   └── codex7_agent/
│       ├── agent/              # Core agent loop implementation
│       │   ├── orchestrator.py # Main orchestrator
│       │   ├── loop.py         # Agent loop primitives
│       │   ├── context_manager.py
│       │   └── state_manager.py
│       ├── tools/              # Tool implementations and management
│       │   ├── tool_manager.py # Tool registry and orchestration
│       │   ├── tool_adapter.py # Adapter for existing tools
│       │   ├── base.py         # Base tool classes
│       │   ├── bash_tool.py
│       │   ├── edit_tool.py
│       │   └── task_tool.py
│       ├── utils/              # Utility functions
│       ├── config/             # Configuration management
│       └── prompt/             # Prompt generation
├── docs/                       # Documentation
└── tests/                      # Test suite
```

### Core Components

1. **Agent Loop Orchestrator**: The central component that manages the agent's execution flow, including intent recognition, tool selection, and result processing.

2. **Tool Management System**: A sophisticated tool registry and orchestration framework that supports both built-in tools and external tool adapters.

3. **Context Manager**: Maintains the conversation context and state across agent iterations.

4. **Intent Recognition**: Analyzes user input to determine task type and complexity.

5. **Prompt Generator**: Dynamically generates system prompts based on the current context and available tools.

## Key Innovations

### 1. Dynamic Agent Loop
The agent implements an intelligent loop that can adapt its behavior based on task complexity and execution results. Unlike traditional linear agent flows, Codex7-Agent can:
- Dynamically adjust the number of iterations
- Modify its approach based on intermediate results
- Handle interruptions and resume execution

### 2. Tool Orchestration System
The tool management system features:
- **Tool Registry**: Centralized tool registration with metadata
- **Tool Adapter Pattern**: Seamless integration of existing tools with the framework
- **Intelligent Scheduling**: Concurrent and sequential tool execution based on safety and priority
- **Force Patterns**: Special syntax to force specific tool usage

### 3. SubAgent Mechanism
For complex tasks, the system employs a SubAgent approach:
- **Task Decomposition**: Automatically breaks down complex tasks into manageable subtasks
- **Parallel Execution**: Executes subtasks in parallel when possible
- **Intelligent Aggregation**: Combines results from multiple SubAgents with conflict resolution

## Usage Flow

1. **Initialization**: 
   - Configure the agent with workspace path, LLM settings, and other parameters
   - Initialize core components (context manager, tool registry, etc.)

2. **Input Processing**:
   - Receive user input or task description
   - Perform intent recognition to classify the task type and complexity

3. **Planning**:
   - Generate system prompt with relevant context and available tools
   - Select appropriate tools based on task requirements
   - For complex tasks, decompose into subtasks for SubAgent processing

4. **Execution**:
   - Execute selected tools either sequentially or in parallel
   - Handle tool results and errors appropriately
   - Update context with execution results

5. **Response Generation**:
   - Generate natural language response based on tool execution results
   - Add response to conversation context

6. **Continuation**:
   - Determine if the task is complete or requires additional iterations
   - Either continue the loop or terminate execution

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd codex7-agent

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Getting Started

```python
from codex7_agent import AgentLoopOrchestrator, AgentConfig

# Create agent configuration
config = AgentConfig(
    workspace_path="./workspace",
    max_loops=50,
    enable_task_tool=True
)

# Initialize the agent
agent = AgentLoopOrchestrator(config)

# Start the agent loop
await agent.start_loop()

# Process a task
result = await agent.process_message("Create a Python function to calculate Fibonacci numbers")
```

## Tools

The agent comes with a comprehensive set of built-in tools:

### File System Tools
- `read_file`: Read content from a file
- `write_file`: Write content to a file
- `list_directory`: List contents of a directory
- `edit_file`: Edit file content with string replacements

### System Tools
- `bash`: Execute bash commands
- `env`: Get environment variables

### Code Tools
- `analyze_code`: Analyze code structure and complexity

### Search Tools
- `grep`: Search for patterns in files
- `find`: Find files by name or pattern

### Task Tool
- `task`: Execute complex tasks with SubAgent orchestration

### Custom Tools
You can easily add custom tools by implementing the `Tool` base class and registering them with the tool registry.

## Configuration

The agent can be configured through:
1. Configuration files (YAML/JSON)
2. Environment variables
3. Direct parameter passing to the `AgentConfig` constructor

Key configuration options include:
- `workspace_path`: Path to the working directory
- `max_loops`: Maximum number of agent iterations
- `enable_task_tool`: Enable/disable the SubAgent mechanism
- `llm_config`: LLM provider and model settings

## Contributing

We welcome contributions to the Codex7-Agent project. Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.