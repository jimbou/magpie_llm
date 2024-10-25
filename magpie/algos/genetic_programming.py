import copy
import math
import random

import magpie.core
import magpie.utils
import os

from .model import get_model
from .llm_call import llm_crossover


class GeneticProgramming(magpie.core.BasicAlgorithm):
    def __init__(self):
        super().__init__()
        self.name = 'Genetic Programming'
        self.config['pop_size'] = 10
        self.config['delete_prob'] = 0.5
        self.config['offspring_elitism'] = 0.5
        self.config['offspring_crossover'] = 0.5
        self.config['offspring_mutation'] = 0.4
        self.config['batch_reset'] = True

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
           
           
            # Write the original program to the file
           
            original_program = self.software

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
            with open("selected_parents_log.txt", "a") as file:
                file.write("Original Program:\n")
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
                parents = self.select(pop)
                # elitism
                copy_parents = copy.deepcopy(parents)
                k = int(self.config['pop_size']*self.config['offspring_elitism'])
                for parent in copy_parents[:k]:
                    offsprings.append(parent)

                #llm 
                # Open the file to log selected parents and their fitness
               # Open the file to log selected parents and their fitness
                parents_string =""
                with open("selected_parents_log.txt", "a") as file:
                    for parent in copy_parents[:k]:
                        # Select 5 random parents from the top K
                        selected_parents = random.sample(copy_parents, 5)

                        # Log each selected parent to the file
                        for i, selected_parent in enumerate(selected_parents):
                            variant = magpie.core.Variant(self.software, selected_parent)
                            fitness = pop[selected_parent].fitness
                            edits = selected_parent.edits  # Assuming edits is a list of Edit objects

                            # Convert each edit to its string representation
                            edits_str = [str(edit) for edit in edits]

                            # Write the parent and its fitness to the file
                            file.write(f"Generation {self.stats['gen']}, Parent {i + 1}:\n")
                            parents_string += f" Parent {i + 1}:\n with fitness {fitness}\n"
                            file.write(f"Fitness: {fitness}\n")
                            file.write(f"Edits: {edits_str}\n")
                            parents_string += f"Parent {i+1} edits: {edits_str}\n"
                            file.write("-----\n")
                        model = get_model("gpt-3.5-turbo-0125",0.7 ,"/home/jim/magpie_llm/llm_logs")
                        response = llm_crossover(parents_string, original_program_info, model)
                        #print the response in the file
                        file.write(f"Response: {response}\n")   

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
