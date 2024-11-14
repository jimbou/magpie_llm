import re
import ast



# This script's responsible for executing small code snippets and determining the resulting program state based on the provided initial state and program code. It is the general script for a simple program statement (not loops or ifs, try etc)
PROMPT1 = """
You have been assigned the role of crossover assistant during genetic programming. You are assisting in the crossover between 2 parents. The goal is to create a child program that is a combination of the 2 parents. You are given multiple possible parents, which are represented as a series of edits. The source file (code program or parameter file)  you are working on is presented below to give you context.
Your task is to select from the available parents, 2 to use as your parents and then from those 2 parents, select the edits you think are the more beneficial to create a child program. The child program must be a combination of the edits from the 2 selected parents. The child must be a combination of the available edits from the 2 selected parents. Your response must adhere to the text format: Child: ***the child***.

{program}
and these are the available parents each with thrie fitness. The lowest fitness the better the parent.
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



def extract_edits(edits_llm: str):
    if '***' in edits_llm:
        edits_llm = edits_llm[:edits_llm.rfind('***')]
        #replace the *** with ''
        edits_llm = edits_llm.replace('***', '')

    if '[' in edits_llm:
        # Format with brackets, we can parse directly as a list
        edit_strings = ast.literal_eval(edits_llm.replace('Child: ', ''))
    else:
        # Format with hyphens, need to split by lines and remove hyphens
        edit_lines = edits_llm.replace("Child:", "").strip().splitlines()
        edit_strings = [line.strip("- ").strip() for line in edit_lines]
    return edit_strings



# This is the main function, it completes the prompt, queries the model and extracts the result, meaining the output state of that program part
def llm_crossover_2parents(parents, program, model, documentation=None):
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


