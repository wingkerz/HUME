# HUME
Collaborative AI Agents Uncover Compact Symbolic Patterns in Human Behaviour Dynamics

## 1. Environment Setup

First, make sure you have installed the required dependencies. The project provides a `yaml` environment configuration file:

```bash
# Create environment using conda
conda env create -f agent_environment.yml

# Activate the environment
conda activate [your_environment_name]
```

---

## 2. Agent Framework

This project is built using **CrewAI**, a multi-agent orchestration framework that enables collaboration and coordination among autonomous AI agents.

For more information about CrewAI, please refer to the [official CrewAI documentation](https://docs.crewai.com/).


## LLM Configuration

This project supports both **open-source** and **closed-source** large language models (LLMs). Users can select and configure different LLM backends according to their computational resources and requirements.

### 1. Open-Source LLMs

Open-source LLMs can be deployed locally using either **Ollama** or **vLLM**.

#### 1.1 Using Ollama

[Ollama](https://ollama.com/) provides a convenient way to download, deploy, and run open-source LLMs locally.

For installation instructions and detailed usage, please refer to the [official Ollama documentation](https://docs.ollama.com/).

**Option 1: Using an older version of Ollama**

For older Ollama configurations, the model can be initialized using `ChatOpenAI`. For example, to load Qwen2.5-7B, use the following configuration in `test_main_HUME_Regressor.py`:

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "NA"

model = ChatOpenAI(model="ollama/qwen2.5:7b")
```

**Option 2: Using a newer version of Ollama**

For newer open-source models, upgrading Ollama may be necessary. For example, with **Ollama v0.34.2**, the model can be configured using CrewAI's `LLM` interface:

```python
from crewai import LLM

model = LLM(
    model="openai/qwen3.8:latest",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    reasoning_effort="none",
)
```

Here, `reasoning_effort="none"` is used to disable reasoning mode when supported by the model and backend.

The model name should be adjusted according to the locally installed Ollama model.

For more information, refer to the [Ollama OpenAI Compatibility documentation](https://docs.ollama.com/api/openai-compatibility).

#### 1.2 Using vLLM

Alternatively, open-source models can be deployed using [vLLM](https://github.com/vllm-project/vllm), which provides an efficient inference engine and an OpenAI-compatible API server.

After downloading the desired model and starting the vLLM service, configure the model as follows:

```python
from crewai import LLM

model = LLM(
    model="llm/qwen3-8b",  # Replace with your local model path or served model name
    base_url="http://localhost:8000/v1",  # Replace with your local vLLM API URL
    api_key="api_key",  # Replace with your local API key
    custom_openai=True,
    temperature=0.9,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False  # Disable reasoning mode
        }
    }
)
```

**Notes:**
- Replace `model` with the model identifier registered with your vLLM server.
- Replace `base_url` with the address of your running vLLM service.
- Replace `api_key` with the API key configured for your local service.
- `custom_openai=True` enables communication through an OpenAI-compatible endpoint in supported CrewAI versions.
- `enable_thinking=False` disables the reasoning mode for supported models, such as compatible Qwen3 models.

For installation and deployment details, please refer to the [official vLLM documentation](https://docs.vllm.ai/).

### 2. Closed-Source LLMs

This framework also supports closed-source LLMs through API-based access, such as models provided by OpenAI.

For example, to use **GPT-3.5-Turbo**, configure the API endpoint and API key as follows:

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "base_url"  # Replace with your API endpoint
os.environ["OPENAI_API_KEY"] = "api_key"  # Replace with your API key

model = ChatOpenAI(model="gpt-3.5-turbo")
```

Replace `base_url` and `api_key` with your own API configuration.

The model name can also be changed to another supported model according to the API provider.

For more information, please refer to the [official OpenAI API documentation](https://platform.openai.com/docs/overview).

### 3. Model Selection

After initializing the desired LLM, assign it to the corresponding agents in the CrewAI framework:

```python
from crewai import Agent

agent = Agent(
    role="Symbolic Regression Expert",
    goal="Discover mathematical expressions from data",
    backstory="An expert in symbolic regression and mathematical modeling.",
    llm=model,
    verbose=True
)
```

Users can switch between different LLM backends by modifying the model initialization configuration without changing the overall multi-agent workflow.


