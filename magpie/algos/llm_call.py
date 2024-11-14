import re
import ast



# This script's responsible for executing small code snippets and determining the resulting program state based on the provided initial state and program code. It is the general script for a simple program statement (not loops or ifs, try etc)
PROMPT1 = """
You have been assigned the role of crossover assistant during genetic programming. You are assisting in the crossover between multiple parents. The goal is to create a child program that is a combination of the multiple parents. The parents are represented as a series of edits. The source file (code program or parameter file)  you are working on is presented below to give you context.
The fitness is runtime, the lower the better.
Your task is to select from the available parents the edits you think are the more beneficial to create a child program. The child program must be a combination of the edits from the available parents. The child must be a combination of the available edits. Your response must adhere to the text format: Child: ***the child***.

{program}
and these are the available parents each with his fitness. The lowest fitness the better the parent.
Available parents:
{parents}
"""
PROMPT2 = """
We are also giving you some useful context about the program you are working on such as documentation:
{documentation}
"""

PROMPT3 = """
Remember you are assisting in the crossover between the parents. Choose as many and whichever edits from the available parents you think will lead to the best child. The proposed edits must be a combination of the available edits. Respond back with the child in the form of a list of edits in the same format as the parents are.
Your response must adhere to the text format: Child: ***the child***. 
"""


# Extracts the postcondition from the model's response
import re

def extract_postcondition(s: str) -> str:
    pattern = r"Postcondition:\s*\*\*(.*?)\*\*"
    matches = re.findall(pattern, s, re.DOTALL)
    if matches:
        # Select the last match
        res = matches[-1]
        # Clean up the beginning and end of the string for any weird characters like * or newlines
        return res.strip()
    return s


# Extracts the result from the model's response given a keyword . For example the keyword can be "Output State"
# Same as extact_postcondition if the keyword is "Postcondition"
def extract_result(s: str, keyword: str) -> str:
    pattern = fr"{keyword}:\s*\*\*(.*?)\*\*"
    matches = re.findall(pattern, s, re.DOTALL)
    if matches:
        # Select the last match
        res = matches[-1]
        # Clean up the beginning and end of the string for any weird characters like * or newlines
        return res.strip()
    return s

def extract_edits(edits_llm: str):
    if '***' in edits_llm:
        edits_llm = edits_llm[:edits_llm.rfind('***')]
        #replace the *** with ''
        edits_llm = edits_llm.replace('***', '')

    if '[' in edits_llm:
        #remove everything before the first '[' and after the last ']'
        edits_llm = edits_llm[edits_llm.find('['):edits_llm.rfind(']')+1]
        return ast.literal_eval(edits_llm)
    else:
        # Format with hyphens, need to split by lines and remove hyphens
        edit_lines = edits_llm.replace("Child:", "").strip().splitlines()
        edit_strings = [line.strip("- ").strip() for line in edit_lines]
    return edit_strings



# This is the main function, it completes the prompt, queries the model and extracts the result, meaining the output state of that program part
def llm_crossover(parents, program, model, documentation=None):
    if program is None:
        program =""
    else :
        program = "The source file\n"+program
    prompt = PROMPT1.format(parents=parents, program=program)
    if documentation:
        prompt += PROMPT2.format(documentation=documentation)
    prompt += PROMPT3
    response = model.query(prompt)
    post = extract_edits(response)
    print("*" * 50)
    print(f"LLM post: {post}")
    return post


