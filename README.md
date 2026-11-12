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








## 2. Model Preparation

Before running the project, you need to configure both the **Embedding model** and the **LLM**.

### Embedding Model

1. Download the pretrained embedding model (For example : https://hf-mirror.com/nvidia/NV-Embed-v2).
2. Place the model files into the `model/` directory.
3. Update the model path in `main.py` and `use_localmodel.py`.

### LLM Configuration

The project supports both **local deployment** and **online API-based models**:

#### Local Open-source Models

1. Download the model files into the `model/` directory.
2. Modify the model loading path in `use_localmodel.py`.

#### Closed-source Models (API)

If you use online models such as GPT:

1. Update the API Endpoint in `use_gpt.py`.
2. Provide your API Key.
3. Replace all occurrences of `use_localmodel` with `use_gpt` in the functions **`gen_use_llm_separate`**, **`suggest_use_llm`**, as well as in the main function.

Modify the `prompts_path` in the main function to your local path, for example:
`prompts_path = "/home/SymBOL/prompts/SR/gen_prompts_2.txt"`

In the prompt, the **Operators list** specifies the operators that may appear in the generated expressions. It can be flexibly configured according to the specific task.
For example:
`{+,-,*,/,^,sqrt,exp,log,sin,cos}` (Low-dimensional tasks)
`{+,-,*,/,^,exp}` (High-dimensional tasks)


---

# Data Preparation

The project provides two data processing pipelines for different application scenarios:

## A. Low-dimensional Symbolic Regression

The project uses the **LSR-transformer dataset** (https://huggingface.co/datasets/nnheui/llm-srbench).

* **Train data**: Pre-generated train datasets are available in the `data/` directory.
* **Switch test cases**: Modify the `case_name` in `screen_pretrain_knowledge_eq.py` to select different examples.
* **Custom datasets**: To test on other datasets, follow the existing format to generate corresponding experimental data.

---

## B. High-dimensional Network Dynamics Symbolic Regression

Taking the **Lotka–Volterra** scenario as an example, network dynamic data is generated via simulation.

### Data Generation

Run the following script to generate data:

```bash
python generate_data.py
```

### Configuration

In the configuration files located in the `configs/Simulation_config.yaml` directory, you can customize:

* Network topology structure
* Integration time steps
* Time range

### Reference Equations

The file `data/dataset_test3.csv` contains the equations corresponding to the **Lotka–Volterra** scenario.

Run the following script to generate data:

```bash
python generate_data.py
```

Run the following script to generate **high-order** data:

 ```bash
python generate_data_high_order.py
 ```

 Run the following script to generate **nonlinear** data:

 ```bash
 python generate_data_noliner.py
 ```

---

# Running Experiments

After configuring the models and preparing the data, run the main script to perform symbolic regression on high-dimensional network dynamics:

```bash
python main_high_SR.py
```
For the high-order scenario, the {variables_list} used to generate g needs to be modified accordingly.
For the nonlinear scenario, the LLM needs to generate s one additional time.
The low-dimensional symbolic regression task can be executed using:
```bash
python main_low_SR.py
```


---
