English | [简体中文](README_zh.md)

# Web OpenManus 🙋

A basic WebUI for OpenManus made using Gradio, visualizing the process of chatting with the UI agent.  
Visit https://github.com/mannaandpoem/OpenManus for the opensource OpenManus repository.

Made by a Chinese high school student, so the function of this UI is still very simple now. Continuing updating with the OpenManus project. 

<img src="img_1.png" width="800px">

## Installation

1. Create a new conda environment:

```bash
conda create -n open_manus python=3.12
conda activate open_manus
```

2. Clone the repository:

```bash
git clone https://github.com/BroWo1/Web-OpenManus.git
cd Web-OpenManus
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

OpenManus requires configuration for the LLM APIs it uses. Follow these steps to set up your configuration:

1. Edit `config/config.toml` to add your API keys and customize settings:

```toml
# Global LLM configuration
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."  # Replace with your actual API key
max_tokens = 4096
temperature = 0.0

# Optional configuration for specific LLM models
[llm.vision]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."  # Replace with your actual API key
```

## Quick Start

One line for run OpenManus:

```bash
python web.py
```

Then open the link in the terminal!

## See Also

Other GitHub repositories of the author:  
https://github.com/BroWo1/GPE-Hub  
https://github.com/Humanoid-a/gpeclubwebsite
