# ChainThought
DS542 Final Project

### Dataset

[GSM8K](https://github.com/openai/grade-school-math)

### Base Model

[Deepseek 7B](https://github.com/deepseek-ai/deepseek-LLM)

### Running the base model evaluation script

#### Running on the SCC

- Start interactive session
- Navigate to the project directory
- Load miniconda and activate Python environment
- Create a virtual environment named `.venv` (since that's what's in the `.gitignore` file)
- Install requirements

Once the requirements have been installed, `run_with_cache.sh` contains all the commands to run the evaluation code for the base model on an SCC interactive session.

#### Extra files for evaluation

`clean_cache.sh` lets users clear the Hugging Face cache stored in the working directory.

`testing.py` lets users test individual math problems by replacing the input sentence and adjusting `max_tokens`.

### Example output

Check the `example_output.json` file for examples of how the model responds to questions from the GSM8K dataset.

### TODO

Update example output with examples of when the `max_tokens` value is set to a larger number (between 1000 and 2000 instead of the current 100). Examples in the example output have been truncated because of the small number of new tokens that can be generated.
