import copy
import math
import random
import ast
import magpie.settings
import magpie.core
import magpie.utils
import os
import re
import math
from .model import get_model
from .llm_call import llm_crossover
from.llm_call_2parents import llm_crossover_2parents


class GeneticProgramming(magpie.core.BasicAlgorithm):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming'
        self.config['pop_size'] = 10
        self.config['delete_prob'] = 0.5
        self.config['offspring_elitism'] = 0.1
        self.config['offspring_crossover'] = 0.5
        self.config['offspring_mutation'] = 0.4
        self.config['batch_reset'] = True
        self.config["llm_multiple_parents"]= magpie.settings.llm_multiple_parents
        self.config["llm_documentation_path"]= magpie.settings.llm_documentation_path
    def reset(self):
        super().reset()
        self.stats['gen'] = 0

    def setup(self, config):
        super().setup(config)
        sec = config['search.gp']
        self.config['pop_size'] = int(sec['pop_size'])
        self.config['delete_prob'] = float(sec['delete_prob'])
        self.config['offspring_elitism'] = float(sec['offspring_elitism'])
        self.config['offspring_crossover'] = float(sec['offspring_crossover'])
        self.config['offspring_mutation'] = float(sec['offspring_mutation'])
        self.config['uniform_rate'] = float(sec['uniform_rate'])
        tmp = sec['batch_reset'].lower()
        if tmp in ['true', 't', '1']:
            self.config['batch_reset'] = True
        elif tmp in ['false', 'f', '0']:
            self.config['batch_reset'] = False
        else:
            msg = '[search.gp] batch_reset should be Boolean'
            raise magpie.core.ScenarioError(msg)

    def aux_log_counter(self):
        gen = self.stats['gen']
        step = self.stats['steps']%self.config['pop_size']+1
        return f'{gen}-{step}'

    def run(self):
        try:
            # warmup
            self.hook_warmup()
            self.warmup()

            # early stop if something went wrong during warmup
            if self.report['stop']:
                return

            # start!
            self.hook_start()

            # initial pop
            pop = {}
            local_best_fitness = None
            while len(pop) < self.config['pop_size']:
                sol = magpie.core.Patch()
                self.mutate(sol)
                if sol in pop:
                    continue
                variant = magpie.core.Variant(self.software, sol)
                run = self.evaluate_variant(variant)
                accept = best = False
                if run.status == 'SUCCESS':
                    if self.dominates(run.fitness, local_best_fitness):
                        local_best_fitness = run.fitness
                        accept = True
                        if self.dominates(run.fitness, self.report['best_fitness']):
                            self.report['best_fitness'] = run.fitness
                            self.report['best_patch'] = sol
                            best = True
                self.hook_evaluation(variant, run, accept, best)
                pop[sol] = run
                self.stats['steps'] += 1

            # main loop
            while not self.stopping_condition():
                self.stats['gen'] += 1
                self.hook_main_loop()
                offsprings = []
                parents = self.select(pop)
                # elitism
                copy_parents = copy.deepcopy(parents)
                k = int(self.config['pop_size']*self.config['offspring_elitism'])
                for parent in copy_parents[:k]:
                    offsprings.append(parent)
                # crossover
                copy_parents = copy.deepcopy(parents)
                k = int(self.config['pop_size']*self.config['offspring_crossover'])
                for parent in copy_parents[:k]:
                    sol = copy.deepcopy(random.sample(parents, 1)[0])
                    if random.random() > 0.5:
                        sol = self.crossover(parent, sol)
                    else:
                        sol = self.crossover(sol, parent)
                    offsprings.append(sol)
                # mutation
                copy_parents = copy.deepcopy(parents)
                k = int(self.config['pop_size']*self.config['offspring_mutation'])
                for parent in copy_parents[:k]:
                    self.mutate(parent)
                    offsprings.append(parent)
                # regrow
                while len(offsprings) < self.config['pop_size']:
                    sol = magpie.core.Patch()
                    self.mutate(sol)
                    if sol in pop:
                        continue
                    offsprings.append(sol)
                # replace
                pop.clear()
                local_best_fitness = None
                for sol in offsprings:
                    if self.stopping_condition():
                        break
                    variant = magpie.core.Variant(self.software, sol)
                    run = self.evaluate_variant(variant)
                    accept = best = False
                    if run.status == 'SUCCESS':
                        if self.dominates(run.fitness, local_best_fitness):
                            local_best_fitness = run.fitness
                            accept = True
                            if self.dominates(run.fitness, self.report['best_fitness']):
                                self.report['best_fitness'] = run.fitness
                                self.report['best_patch'] = sol
                                best = True
                    self.hook_evaluation(variant, run, accept, best)
                    pop[sol] = run
                    self.stats['steps'] += 1

        except KeyboardInterrupt:
            self.report['stop'] = 'keyboard interrupt'

        finally:
            # the end
            self.hook_end()

    def mutate(self, patch):
        if patch.edits and random.random() < self.config['delete_prob']:
            del patch.edits[random.randrange(0, len(patch.edits))]
        else:
            patch.edits.append(self.create_edit(self.software.noop_variant))

    def crossover(self, sol1, sol2):
        c = copy.deepcopy(sol1)
        for edit in sol2.edits:
            c.edits.append(edit)
        return c

    def filter(self, pop):
        return {sol for sol in pop if pop[sol].status == 'SUCCESS'}

    def select(self, pop):
        """ returns possible parents ordered by fitness """
        return sorted(self.filter(pop), key=lambda sol: pop[sol].fitness)

    def hook_main_loop(self):
        if self.config['batch_reset']:
            for a in self.config['batch_bins']:
                random.shuffle(a)
            self.hook_reset_batch()

    def create_edits_from_llm_response(self,response, variant=None):
    
        ref = variant or self.software.noop_variant
        edits = []
        print(f"the response is {response}")
        for edit_str in response:
            # Parse the edit type and arguments
            print(f"current edit string is {edit_str}")
            match = re.match(r"(\w+)\((.*)\)", edit_str)
            if not match:
                print(f"Error: Could not parse edit string '{edit_str}'")
                continue

            edit_type, args_str = match.groups()

            # Find the class in possible edits
            klass = next((cls for cls in self.config['possible_edits'] if cls.__name__ == edit_type), None)
            if klass is None:
                print(f"Error: Edit type '{edit_type}' not found in possible edits.")
                continue

            # Create the Edit object with parsed target and additional args
            args = ast.literal_eval(f"({args_str})")
            target = args[0]  # Assuming the first argument is always the target
            additional_args = args[1:]

            # Direct instantiation instead of auto_create if arguments are sufficient
            edit_obj = klass(target, *additional_args)
            edits.append(edit_obj)
            print(f"Edit object created: {edit_obj}")
        
        return edits



class GeneticProgrammingConcat(GeneticProgramming):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming (Concat)'

    def crossover(self, sol1, sol2):
        c = copy.deepcopy(sol1)
        for edit in sol2.edits:
            c.edits.append(edit)
        return c

magpie.utils.known_algos.append(GeneticProgrammingConcat)

class GeneticProgrammingLLM(GeneticProgramming):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming (LLM)'

    def run(self):
        try:
            # warmup
            self.hook_warmup()
            self.warmup()

            # early stop if something went wrong during warmup
            if self.report['stop']:
                return

            # start!
            self.hook_start()

            # initial pop
            pop = {}
            local_best_fitness = None
            #read the contents of the file we are chnaging as they are stored in original_program.target_files
            target_file_contents = {}

            
           
            # Write the original program to the file
           
            original_program = self.software
            
            for target_file in original_program.target_files:
                try:
                    #keep only the last file in the path of target_file
                    target_file = target_file.split("/")[-1]
                    original_program_path =  original_program.config['software']['path']+"/"+target_file
                    with open(original_program_path, 'r') as file:
                        contents = file.read()
                        target_file_contents[target_file] = contents  # Store contents with the file path as the key
                        # print(f"Contents of {target_file}:\n{contents}\n")
                except FileNotFoundError:
                    print(f"Error: {original_program_path} does not exist.")
                except Exception as e:
                    print(f"Error reading {target_file}: {e}")

            # tranform target_file_contents to string
            target_file_contents_str = ""
            for target_file, contents in target_file_contents.items():
                target_file_contents_str += f"{contents}\n"

            if len(target_file_contents_str)> 1000: #too big of a file
                target_file_contents_str = None
            

            # Collect information from the BasicSoftware instance
            original_program_info = f"Path: {original_program.config['software']['path']}\n"
            original_program_info += f"Target Files: {', '.join(original_program.target_files)}\n"

            # Add model rules if available
            if original_program.model_rules:
                original_program_info += "Model Rules:\n"
                for k, v in original_program.model_rules:
                    original_program_info += f"  {k}: {v}\n"

            # Add model config if available
            if original_program.model_config:
                original_program_info += "Model Config:\n"
                for k, v in original_program.model_config:
                    original_program_info += f"  {k}: {v}\n"

            # Add fitness type
            original_program_info += f"Fitness Type: {original_program.fitness_type}\n"

            # Add commands and their timeouts
            if original_program.init_cmd:
                original_program_info += f"Init Command: {original_program.init_cmd}, Timeout: {original_program.init_timeout}\n"
            if original_program.setup_cmd:
                original_program_info += f"Setup Command: {original_program.setup_cmd}, Timeout: {original_program.setup_timeout}\n"
            if original_program.compile_cmd:
                original_program_info += f"Compile Command: {original_program.compile_cmd}, Timeout: {original_program.compile_timeout}\n"
            if original_program.test_cmd:
                original_program_info += f"Test Command: {original_program.test_cmd}, Timeout: {original_program.test_timeout}\n"
            if original_program.run_cmd:
                original_program_info += f"Run Command: {original_program.run_cmd}, Timeout: {original_program.run_timeout}\n"

            # Add batch parameters
            original_program_info += f"Batch Fitness Strategy: {original_program.batch_fitness_strategy}\n"
            original_program_info += f"Batch Bin Strategy: {original_program.batch_bin_fitness_strategy}\n"

            # Write the original program information to the file
            with open("selected_parents_log.txt", "w") as file:
                file.write(original_program_info)
                file.write("\n-----\n")

                # Pretty print the contents of each target file
                for target_file in original_program.target_files:
                    target_file_path = os.path.join(original_program.config['software']['path'], target_file)
                    if os.path.exists(target_file_path):
                        file.write(f"Contents of {target_file}:\n")
                        file.write("-----\n")
                        original_program_info += f"Contents of {target_file}:\n"
                        original_program_info += "-----\n"
                        with open(target_file_path, "r") as f:
                            contents = f.read()
                            file.write(contents)
                            file.write("\n-----\n")
                            original_program_info += contents
                            original_program_info += "\n-----\n"
                    else:
                        file.write(f"Error: {target_file} does not exist at path {target_file_path}\n")
                        file.write("-----\n")

            # Finished writing the original program to the file

            while len(pop) < self.config['pop_size']:
                sol = magpie.core.Patch()
                self.mutate(sol)
                if sol in pop:
                    continue
                variant = magpie.core.Variant(self.software, sol)
                run = self.evaluate_variant(variant)
                accept = best = False
                if run.status == 'SUCCESS':
                    if self.dominates(run.fitness, local_best_fitness):
                        local_best_fitness = run.fitness
                        accept = True
                        if self.dominates(run.fitness, self.report['best_fitness']):
                            self.report['best_fitness'] = run.fitness
                            self.report['best_patch'] = sol
                            best = True
                self.hook_evaluation(variant, run, accept, best)
                pop[sol] = run
                self.stats['steps'] += 1

            # main loop
            while not self.stopping_condition():
                self.stats['gen'] += 1
                self.hook_main_loop()
                offsprings = []
                tries=0 #num of llm calls
                parents = self.select(pop)
                # elitism
                copy_parents = copy.deepcopy(parents)
                #sort the parents by fitness
                copy_parents = sorted(copy_parents, key=lambda sol: pop[sol].fitness)
        
                k1 = int(self.config['pop_size']*self.config['offspring_elitism'])
                for parent in copy_parents[:k1]:
                    offsprings.append(parent)
                #take the best in the next generation
                if k1< 5:
                    k= 2*k1
                else:
                    k = 5
                print(f"k is {k}")
                print(f"pop size is {self.config['pop_size']}")
                offsprings_from_crossover =self.config['pop_size'] *self.config['offspring_crossover']
                # for parent in copy_parents[:k]:
                #     offsprings.append(parent)

                #llm 
                # Open the file to log selected parents and their fitness
               # Open the file to log selected parents and their fitness
                parents_string =""
                #the maximum number of llm calls will be the min of either twice the number of chiodren we want to create from crossover or the number of combinations of parents
                num_combinations = math.comb(len(copy_parents), 2)
                max_tries = min(2*offsprings_from_crossover, num_combinations)
                with open("selected_parents_log.txt", "a") as file:

                    while len(offsprings) < offsprings_from_crossover and tries < max_tries:
                        parents_string =""
                        #get the k best parents from the population or if parents less than k get them all
                        if len(copy_parents) > k:
                            print(f"i am selecting the best {k} parents")
                            best_parents = copy_parents[:k]
                        else:
                            best_parents = copy_parents


                        
                            # Select 5 random parents from the top K
                            #if K larger than 5 select 5 parents
                        if len(best_parents) > 5:
                            print("Selecting 5 random parents from the top 5")
                            selected_parents = random.sample(best_parents, 5)
                        else:
                            selected_parents = best_parents

                        print("Selected parents: ", selected_parents)
                        print("all parents: ", copy_parents)

                        #crossover until you get 
                        # Log each selected parent to the file
                        crossover_num =1
                        edit_dict = {}

                        for selected_parent in selected_parents:
                            # variant = magpie.core.Variant(self.software, selected_parent)
                            fitness = pop[selected_parent].fitness
                            edits = selected_parent.edits  # Assuming edits is a list of Edit objects

                            edits_str=[]
                            # Convert each edit to its string representation
                            for edit in edits:
                                edit_str = str(edit)
                                edit_dict[edit_str] = edit 
                                edits_str.append(edit_str)
                            edits_count = len(edits)
                            # Write the parent and its fitness to the file
                            file.write(f"Generation {self.stats['gen']}, Parent {crossover_num}:\n")
                            parents_string += f" Parent {crossover_num}:\n with fitness {fitness}\n"
                            file.write(f"Fitness: {fitness}\n")
                            file.write(f"Edits: {edits_str}\n")
                            parents_string += f"Parent {crossover_num} has {edits_count} edits: {edits_str}\n"
                            file.write("-----\n")
                            crossover_num+=1

                        model = get_model("gpt-4o-mini-2024-07-18",0.7 ,"llm_logs")
                        if self.config["llm_multiple_parents"]:
                        
                            if self.config["llm_documentation_path"].strip() != "":
                                documentation = open(self.config["llm_documentation_path"], "r").read()
                                response = llm_crossover(parents_string, target_file_contents_str, model, documentation=documentation)
                            else:
                                response = llm_crossover(parents_string, target_file_contents_str, model)
                        else:   
                            if self.config["llm_documentation_path"].strip() != "":
                                documentation = open(self.config["llm_documentation_path"], "r").read()
                                response = llm_crossover_2parents(parents_string, target_file_contents_str, model, documentation=documentation)
                            else:
                                response = llm_crossover_2parents(parents_string, target_file_contents_str, model)
                        #print the response in the file
                        file.write(f"Response: {response}\n")  

                        
                            
                           

                        new_edits = []
                        missing_edit = False

                        for edit_str in response:
                            if edit_str in edit_dict:
                                new_edits.append(edit_dict[edit_str])
                            else:
                                print(f"Warning: Edit '{edit_str}' not found in edit dictionary. Skipping this offspring.")
                                missing_edit = True
                                break

                        # If any edit is missing, skip this offspring and continue to the next LLM call
                        if missing_edit:
                            tries+=1
                            # Optionally log that we are skipping the creation of this offspring
                            print(f"Skipping offspring creation due to missing edits in response.\n")
                            continue  # Go to the next LLM call

                        #Convert each parsed edit string to an Edit object
                        # new_edits = []
                        # for edit_str in response:
                        #     print("Edit string: ", edit_str)
                        #     # Extract the class name and arguments using eval on the inner content
                        #     class_name, args_str = edit_str.split("(", 1)
                        #     edit_class = magpie.utils.edit_from_string(class_name)
                        #     args = ast.literal_eval(f"({args_str[:-1]})")  # safely parse arguments

                        #     # Create the Edit object and add to new_edits
                        #     new_edits.append(edit_class(*args))
                        offspring = copy.deepcopy(random.sample(selected_parents, 1)[0])
                        offspring.edits = new_edits  # Replace with LLM-selected edits
                        # print("New edits assigned to offspring:", [str(edit) for edit in offspring.edits])

                        # Add offspring to the population or any other desired processing
                       

                        #i want to check if the solution is viable
                        try:
                            variant = magpie.core.Variant(self.software, magpie.core.Patch(edits=new_edits))
                            offsprings.append(offspring)
                            print("Solution viable at try number: ", tries)
                            tries+=1
                        except Exception as e:
                            print("Solution not viable try again with try number: ", tries) 
                            print(f"the edits are {response}")
                            print(f"the error is {e}")
                            tries+=1
                            continue    
                        

                # crossover
                # copy_parents = copy.deepcopy(parents)
                # k = int(self.config['pop_size']*self.config['offspring_crossover'])
                # for parent in copy_parents[:k]:
                #     sol = copy.deepcopy(random.sample(parents, 1)[0])
                #     if random.random() > 0.5:
                #         sol = self.crossover(parent, sol)
                #     else:
                #         sol = self.crossover(sol, parent)
                #     offsprings.append(sol)
                # mutation
                copy_parents = copy.deepcopy(parents)
                k = int(self.config['pop_size']*self.config['offspring_mutation'])
                for parent in copy_parents[:k]:
                    self.mutate(parent)
                    offsprings.append(parent)
                # regrow
                while len(offsprings) < self.config['pop_size']:
                    sol = magpie.core.Patch()
                    self.mutate(sol)
                    if sol in pop:
                        continue
                    offsprings.append(sol)
                # replace
                pop.clear()
                local_best_fitness = None
                for sol in offsprings:
                    if self.stopping_condition():
                        break
                    variant = magpie.core.Variant(self.software, sol)
                    run = self.evaluate_variant(variant)
                    accept = best = False
                    if run.status == 'SUCCESS':
                        if self.dominates(run.fitness, local_best_fitness):
                            local_best_fitness = run.fitness
                            accept = True
                            if self.dominates(run.fitness, self.report['best_fitness']):
                                self.report['best_fitness'] = run.fitness
                                self.report['best_patch'] = sol
                                best = True
                    self.hook_evaluation(variant, run, accept, best)
                    pop[sol] = run
                    self.stats['steps'] += 1

        except KeyboardInterrupt:
            self.report['stop'] = 'keyboard interrupt'

        finally:
            # the end
            self.hook_end()

magpie.utils.known_algos.append(GeneticProgrammingLLM)


class GeneticProgramming1Point(GeneticProgramming):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming (1-point)'

    def crossover(self, sol1, sol2):
        c = magpie.core.Patch()
        k1 = random.randint(0, len(sol1.edits))
        k2 = random.randint(0, len(sol2.edits))
        for edit in sol1.edits[:k1]:
            c.edits.append(edit)
        for edit in sol2.edits[k2:]:
            c.edits.append(edit)
        return c

magpie.utils.known_algos.append(GeneticProgramming1Point)


class GeneticProgramming2Point(GeneticProgramming):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming (2-point)'

    def crossover(self, sol1, sol2):
        c = magpie.core.Patch()
        k1 = random.randint(0, len(sol1.edits))
        k2 = random.randint(0, len(sol1.edits))
        k3 = random.randint(0, len(sol2.edits))
        k4 = random.randint(0, len(sol2.edits))
        for edit in sol1.edits[:min(k1, k2)]:
            c.edits.append(edit)
        for edit in sol2.edits[min(k3, k4):max(k3, k4)]:
            c.edits.append(edit)
        for edit in sol1.edits[max(k1, k2):]:
            c.edits.append(edit)
        return c

magpie.utils.known_algos.append(GeneticProgramming2Point)


class GeneticProgrammingUniformConcat(GeneticProgramming):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming (uniform+concatenation)'
        self.config['uniform_rate'] = 0.5

    def crossover(self, sol1, sol2):
        c = magpie.core.Patch()
        for edit in sol1.edits:
            if random.random() > self.config['uniform_rate']:
                c.edits.append(edit)
        for edit in sol2.edits:
            if random.random() > self.config['uniform_rate']:
                c.edits.append(edit)
        if len(c.edits) == 0:
            sol3, sol4 = [sol1, sol2] if random.random() > 0.5 else [sol2, sol1]
            if sol3.edits:
                c.edits.append(random.choice(sol3.edits))
            elif sol4.edits:
                c.edits.append(random.choice(sol4.edits))
        return c

magpie.utils.known_algos.append(GeneticProgrammingUniformConcat)


class GeneticProgrammingUniformInter(GeneticProgramming):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming (uniform+interleaved)'
        self.config['uniform_rate'] = 0.5

    def crossover(self, sol1, sol2):
        c = magpie.core.Patch()
        l1 = [(i/len(sol1.edits), 0) for i in sorted(random.sample(range(len(sol1.edits)), math.ceil(len(sol1.edits)*self.config['uniform_rate'])))]
        l2 = [(i/len(sol2.edits), 1) for i in sorted(random.sample(range(len(sol2.edits)), math.ceil(len(sol2.edits)*self.config['uniform_rate'])))]
        for (x, k) in sorted(l1+l2):
            sol = [sol1, sol2][k]
            edit = sol.edits[int(x*len(sol.edits))]
            c.edits.append(edit)
        if len(c.edits) == 0:
            sol3, sol4 = [sol1, sol2] if random.random() > 0.5 else [sol2, sol1]
            if sol3.edits:
                c.edits.append(random.choice(sol3.edits))
            elif sol4.edits:
                c.edits.append(random.choice(sol4.edits))
        return c

magpie.utils.known_algos.append(GeneticProgrammingUniformInter)
