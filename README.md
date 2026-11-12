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

## Dataset Configuration

Before running the code, users need to configure the local project paths and select the dataset for the experiment.

### 1. Configure the Project Root Path

First, open `test_main_HUME_Regressor.py` and modify `project_root` to match the local project directory:

```python
project_root = '/home/ai_agent_local_llm/src'
```

Replace the path above with the corresponding absolute path on your machine.

### 2. Select the Dataset

The `data/` directory contains several datasets used in our experiments.

To select a dataset, open:

`src/ai_agent_local_llm/tools/screen_pretrain_knowledge_eq.py`

Locate the following argument in the parser configuration:

```python
parser.add_argument('--case_name', help='case_name', default="lichess")
```

Modify the `default` value to specify the dataset or experimental case you want to run.

For example, to run the **I.24.6_1_1** case from the **LSR-Transform** benchmark, change it to:

```python
parser.add_argument('--case_name', help='case_name', default="I.24.6_1_1")
```

### 3. Configure Dataset Paths

The dataset loading paths must also be updated according to your local directory structure.

In `src/ai_agent_local_llm/tools/screen_pretrain_knowledge_eq.py`, locate the `load_data_SR` function and modify the following path:

```python
test_data_set_path = 'data/id/test_data_on_%s%s.pickle' % (case_name, add_str)
```

Replace it with the actual path where the dataset is stored on your machine.

Similarly, update the dataset paths in the following functions:

- `load_data_SR`
- `load_data_SR_all_following_cars`
- `load_data_SR_from_csv`

Ensure that the specified paths and data formats are consistent with the corresponding datasets before running the experiments.

### 4. Dataset Description

The `data/` directory contains datasets covering different human behavioral scenarios, including pedestrian dynamics, vehicle-following behavior, chess decision-making, and communication activities.

| Dataset / Directory | Description |
|---|---|
| `Juelich_NPY_Data_9` | Pedestrian dynamics data |
| `npy_pairs_n15_road1` | Vehicle platoon following data used to investigate stop-and-go wave phenomena |
| `lichess` | Chess decision-making data containing thinking-time intervals from games with 3-, 10-, and 30-minute time controls |
| `phone` | Human communication data |
| `agent` | AI agent communication data, including time intervals between agent posts |
| `NGSIM_dataNpy` | Vehicle trajectory data for car-following behavior |
| `LSR-Transform` | Symbolic regression benchmark with transformed mathematical expressions |

#### 4.1 NGSIM Car-Following Dataset

For the NGSIM car-following experiments, the trajectory data are organized into the following directories:

```python
file_folder_list = [
    "data/NGSIM_dataNpy/trajectories-0750am-0805am",
    "data/NGSIM_dataNpy/trajectories-0805am-0820am",
    "data/NGSIM_dataNpy/trajectories-0820am-0835am"
]
```

The corresponding dataset and preprocessing resources can be found in the following repository:

[NGSIM Dataset — SR-LLM](https://github.com/ThuOneLab/SR-LLM/tree/main/data/NGSIM_dataNpy)

Please download the required data and update `file_folder_list` according to your local storage paths.

#### 4.2 LSR-Transform And Others Benchmark

The **LSR-Transform** and other symbolic regression benchmark datasets, including **Nguyen, Nguyen-c, Keijzer, Constant, and R**, are used to evaluate the performance of symbolic regression methods across diverse mathematical expressions.

The LSR-Transform dataset is available from Hugging Face:

[LSR-Transform — LLM-SRBench](https://huggingface.co/datasets/nnheui/llm-srbench/viewer/default/lsr_transform)

To use these benchmark datasets in our framework:

1. Download or generate the required benchmark data.
2. Organize the data into the input and target formats required by the symbolic regression task.
3. Serialize the processed data into Python `pickle` files compatible with the data-loading functions.
4. Update `test_data_set_path` in `load_data_SR` to point to the corresponding local files.
5. Set `case_name` to the desired benchmark case, such as `I.24.6_1_1` for LSR-Transform.

**Note:** All benchmark datasets, including LSR-Transform, Nguyen, Nguyen-c, Keijzer, Constant, and R, must be prepared and converted into the expected `pickle` format before being loaded by the current implementation.

### 5. Running Experiments

After configuring the project root, selecting the desired dataset, and updating the corresponding data paths, the experiment can be launched using:

```bash
python test_main_HUME_Regressor.py
```

Make sure that the selected dataset is available locally and that the LLM backend has been configured correctly before running the experiment.

Users can switch between different LLM backends by modifying the model initialization configuration without changing the overall multi-agent workflow.


