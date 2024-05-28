<h1 align="center">
<br/>
Conveyor
<br/>
</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-3.11-green"/>
<img src="https://img.shields.io/badge/pytorch-2.1-green"/>
<img src="https://img.shields.io/badge/license-apache_2.0-blue"/>
</p>

---

Conveyor is a LLM-serving runtime with efficient tool usage capability. 
Developers can simply create their toolswith a few lines of code to enable partial execution for tools.

## Getting Started

### Set Up Environment

```bash
conda env create -f environment.yml
conda activate conveyor
pip install flashinfer -i https://flashinfer.ai/whl/cu121/
```

### Run Examples

Use `task.py` to run pre-defined examples.

### Create Your Own Tools
Simply inherit `BasePlugin` in `conveyor/plugin/base_plugin.py` and implement corresponding methods. Here is a simple example:
```python
class SumPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.val = 0

    def process_new_dat(self, data):
        # assume your parser gives you some numbers
        self.val += data

    def finish(self):
        return self.val
```
More examples can be found under the same directory.

## License 

Conveyor is licensed under Apache-2.0 license.