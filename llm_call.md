
# LLM-Assisted Crossover in Genetic Programming

## Project Overview

This project demonstrates a proof of concept for using Large Language Models  to assist in the crossover process of genetic programming, particularly focused on code and parameter improvement. The goal is to enhance the quality of generated variants by leveraging LLMs to select and combine beneficial edits from multiple parent solutions. By utilizing an LLM, we aim to improve the accuracy, functionality, and effectiveness of crossover processes beyond traditional random or uniform selection methods.

This README outlines the current implementation, explains each component, and describes the next steps for expanding the project.

## Current Implementation

The code is designed to:
1. **Collect Parent Information**: Selects parents and retrieves relevant information (fitness, edits applied, configuration).
2. **Invoke LLM for Crossover**: Uses an LLM to combine the edits of the selected parents into a new variant.
3. **Create New Variants**: Generates new solution candidates based on the LLM’s response, which suggests a list of edits that could enhance the program's fitness.

### Key Components and Process

#### 1. Prompt Construction

The LLM prompt (`PROMPT`) provides context to the model about the source code and available parents. It requests the model to select and propose edits from the parents to generate a new child variant.

- **Source Program Context**: Supplies basic information about the program being optimized, including target files, configuration details, and relevant commands.
- **Parent Edits and Fitness Values**: Lists the selected parents, including their fitness scores and applied edits, allowing the LLM to make informed selections.
  
The prompt explicitly instructs the LLM to propose a combination of edits from the parents, with the response format specified for easy parsing.

#### 2. LLM Crossover Execution (`llm_crossover` function)

The `llm_crossover` function:
- Formats the prompt with parent and program details.
- Queries the LLM and retrieves the response, ensuring it matches the expected format (either a list with brackets or hyphenated lines).
- Converts the response into a list of edit strings.

#### 3. Extracting and Applying Edits

The `extract_edits` function parses the LLM’s response, accommodating different formats:
- If the response contains brackets (`[]`), it parses it as a Python list.
- If the response contains hyphens (`-`), it splits each line, strips the hyphens, and processes each edit individually.

The extracted edit strings are then transformed into `Edit` objects using the `edit_from_string` utility, ensuring compatibility with the program’s existing editing mechanisms.

### Code Example: LLM-Assisted Crossover Loop

After selecting top parents based on fitness, each set of selected parents is passed to the LLM, which returns a list of edits to apply. These edits are converted to `Edit` objects and stored in the new variant, which is then evaluated and added to the population.

```python
# Sample LLM Crossover Call
response = llm_crossover(parents_string, original_program_info, model)
new_edits = []
for edit_str in response:
    # Create Edit object from each edit string
    class_name, args_str = edit_str.split("(", 1)
    edit_class = magpie.utils.edit_from_string(class_name)
    args = ast.literal_eval(f"({args_str[:-1]})")
    new_edits.append(edit_class(*args))
# Apply edits to new variant
sol.edits = new_edits
offsprings.append(sol)
```

## Next Steps

To further develop the project, consider the following tasks:

1. **Use LLM Exclusively for Crossover**: Remove traditional crossover methods and rely solely on the LLM to produce new children. Adjust the prompt and model configurations as necessary to optimize this process.

2. **Refine the Prompt and Response Handling**:
   - Experiment with prompt adjustments to achieve better selection accuracy and variety in the LLM’s proposed edits.
   - Enhance `extract_edits` to handle other potential response formats and improve robustness.

3. **Introduce LLM-Driven Mutation**:
   - Investigate the potential of LLMs to introduce new edits or mutations in addition to crossover. This could include requesting novel edit suggestions or slight modifications in parameters for further optimization.

4. **Run Experimental Evaluations**:
   - Conduct experiments using the Magpie framework and benchmark against traditional crossover techniques to evaluate improvements in runtime, accuracy, and variant functionality.
   - Document experimental results and analyze the impact of LLM-assisted crossover on fitness improvements and error reduction.


## Running an example
You can read the rest of the available documentation on how to invoke magpie and use scenarios files but for the sake of speed:
```bash
python3.11 magpie genetic_programming --scenario examples/minisat/_magpie/scenario_runtime_config1_time.txt 
```

## New Scenario Parameters for LLM-Assisted Crossover
Two new parameters have been introduced for the scenario file, allowing greater flexibility and control over the LLM-assisted crossover process:

1. **llm_documentation_path:** This parameter specifies the path to a documentation file that provides additional context to the LLM during crossover operations. By default, this is set to None. If a file path is provided, the contents of the specified file will be included as part of the prompt, enriching the context for the LLM and potentially improving the quality of suggested edits. If left unset, the LLM operates without this additional documentation.

2. **llm_multiple_parents:** This boolean parameter (default: True) controls how many parents are used for crossover selection. When True, the LLM considers all provided parents to select from a broader range of mutations. When set to False, only two parents are selected from the available options, allowing a more focused crossover. Adjusting this setting can be useful for testing different crossover strategies or limiting computational resources.