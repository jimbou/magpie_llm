import contextlib
import pathlib
import re
import shlex

import magpie.settings
import magpie.utils.known

from .abstract_software import AbstractSoftware
from .errors import ScenarioError
from .runresult import RunResult



class BasicSoftware(AbstractSoftware):
    def __init__(self, config):
        self.config = config

        # AbstractSoftware *requires* a path, a list of target files, and a list of possible edits
        if not (val := config['software']['path']):
            msg = 'Invalid config file: "[software] path" must be defined'
            raise ScenarioError(msg)
        super().__init__(val, reset=False)
        if not (val := config['software']['target_files']):
            msg = 'Invalid config file: "[software] target_files" must defined'
            raise ScenarioError(msg)
        self.target_files = val.split()

        # model rules
        self.model_rules = []
        try:
            for rule in config['software']['model_rules'].split('\n'):
                if rule: # discard potential initial empty line
                    k, v = rule.split(':')
                    self.model_rules.append((k.strip(), v.strip()))
        except ValueError as e:
            msg = f'Badly formated rule: "{rule}"'
            raise ScenarioError(msg) from e

        # model config
        self.model_config = []
        try:
            for rule in config['software']['model_config'].split('\n'):
                if rule: # discard potential initial empty line
                    k, v = rule.split(':')
                    v = v.strip()
                    if v[0]+v[-1] != '[]':
                        msg = f'Badly formated section name: "{rule}"'
                        raise ScenarioError(msg)
                    self.model_config.append((k.strip(), v[1:-1]))
        except ValueError as e:
            msg = f'Badly formated rule: "{rule}"'
            raise ScenarioError(msg) from e

        # fitness type
        if 'fitness' not in config['software']:
            msg = 'Invalid config file: "[software] fitness" must be defined'
            raise ScenarioError(msg)
        known_fitness = ['output', 'time', 'posix_time', 'perf_time', 'perf_instructions', 'repair', 'bloat_lines', 'bloat_words', 'bloat_chars', 'perf_cycles',"perf_cache_references", "perf_cache_misses", "perf_branches","perf_branch_misses",  "perf_cpu_clock", "perf_task_clock",  "perf_faults", "perf_minor_faults", "perf_major_faults", "perf_cs", "perf_migrations", "perf_L1_dcache_loads","perf_L1_dcache_load_misses", "perf_dTLB_loads", "perf_dTLB_load_misses", "weights","energy", "energy_ram", "energy_uncore"]
        if config['software']['fitness'] not in known_fitness:
            tmp = '/'.join(known_fitness)
            msg = f'Invalid config file: "[software] fitness" key must be {tmp}'
            raise ScenarioError(msg)
        self.fitness_type = config['software']['fitness']

        # execution-related parameters
        self.init_performed = False
        self.init_cmd = None
        self.init_timeout = None
        self.setup_performed = False
        self.setup_cmd = None
        self.setup_timeout = None
        self.setup_lengthout = None
        self.compile_cmd = None
        self.compile_timeout = None
        self.compile_lengthout = None
        self.test_cmd = None
        self.test_timeout = None
        self.test_lengthout = None
        self.run_cmd = None
        self.run_timeout = None
        self.run_lengthout = None
        self.batch_timeout = None
        self.batch_lengthout = None
        self.retries = None

        # init
        if 'init_cmd' in config['software']:
            if config['software']['init_cmd'].lower() in ['', 'none']:
                self.init_cmd = None
            else:
                self.init_cmd = config['software']['init_cmd']
        if 'init_timeout' in config['software']:
            if config['software']['init_timeout'].lower() in ['', 'none']:
                self.init_timeout = None
            else:
                self.init_timeout = float(config['software']['init_timeout'])
        if 'init_lengthout' in config['software']:
            if config['software']['init_lengthout'].lower() in ['', 'none']:
                self.init_lengthout = None
            else:
                self.init_lengthout = int(config['software']['init_lengthout'])

        # setup
        if 'setup_cmd' in config['software']:
            if config['software']['setup_cmd'].lower() in ['', 'none']:
                self.setup_cmd = None
            else:
                self.setup_cmd = config['software']['setup_cmd']
        if 'setup_timeout' in config['software']:
            if config['software']['setup_timeout'].lower() in ['', 'none']:
                self.setup_timeout = None
            else:
                self.setup_timeout = float(config['software']['setup_timeout'])
        if 'setup_lengthout' in config['software']:
            if config['software']['setup_lengthout'].lower() in ['', 'none']:
                self.setup_lengthout = None
            else:
                self.setup_lengthout = int(config['software']['setup_lengthout'])

        # compile
        if 'compile_cmd' in config['software']:
            if config['software']['compile_cmd'].lower() in ['', 'none']:
                self.compile_cmd = None
            else:
                self.compile_cmd = config['software']['compile_cmd']
        if 'compile_timeout' in config['software']:
            if config['software']['compile_timeout'].lower() in ['', 'none']:
                self.compile_timeout = None
            else:
                self.compile_timeout = float(config['software']['compile_timeout'])
        if 'compile_lengthout' in config['software']:
            if config['software']['compile_lengthout'].lower() in ['', 'none']:
                self.compile_lengthout = None
            else:
                self.compile_lengthout = int(config['software']['compile_lengthout'])

        # test
        if 'test_cmd' in config['software']:
            if config['software']['test_cmd'].lower() in ['', 'none']:
                self.test_cmd = None
            else:
                self.test_cmd = config['software']['test_cmd']
        if 'test_timeout' in config['software']:
            if config['software']['test_timeout'].lower() in ['', 'none']:
                self.test_timeout = None
            else:
                self.test_timeout = float(config['software']['test_timeout'])
        if 'test_lengthout' in config['software']:
            if config['software']['test_lengthout'].lower() in ['', 'none']:
                self.test_lengthout = None
            else:
                self.test_lengthout = int(config['software']['test_lengthout'])

        # run
        if 'run_cmd' in config['software']:
            if config['software']['run_cmd'].lower() in ['', 'none']:
                self.run_cmd = None
            else:
                self.run_cmd = config['software']['run_cmd']
        if 'run_timeout' in config['software']:
            if config['software']['run_timeout'].lower() in ['', 'none']:
                self.run_timeout = None
            else:
                self.run_timeout = float(config['software']['run_timeout'])
        if 'run_lengthout' in config['software']:
            if config['software']['run_lengthout'].lower() in ['', 'none']:
                self.run_lengthout = None
            else:
                self.run_lengthout = int(config['software']['run_lengthout'])
        if 'retries' in config['software']:
            if config['software']['retries'].lower() in ['', 'none']:
                self.retries = None
            else:
                self.retries = int(config['software']['retries'])

        # batch parameters
        self.batch = [''] # default initial batch: single empty instance
        self.batch_fitness_strategy = config['software']['batch_fitness_strategy']
        known_strategies = ['sum', 'average', 'median']
        if self.batch_fitness_strategy not in known_strategies:
            tmp = '/'.join(known_strategies)
            msg = f'Invalid config file: "[software] batch_fitness_strategy" key must be {tmp}'
            raise ScenarioError(msg)
        self.batch_bin_fitness_strategy = config['software']['batch_fitness_strategy']
        known_strategies = ['aggregate', 'sum', 'average', 'median', 'q10', 'q25', 'q75', 'q90']
        if self.batch_fitness_strategy not in known_strategies:
            tmp = '/'.join(known_strategies)
            msg = f'Invalid config file: "[software] batch_bin_fitness_strategy" key must be {tmp}'
            raise ScenarioError(msg)
        if 'batch_timeout' in config['software']:
            if config['software']['batch_timeout'].lower() in ['', 'none']:
                self.batch_timeout = None
            else:
                self.batch_timeout = float(config['software']['batch_timeout'])
        if 'batch_lengthout' in config['software']:
            if config['software']['batch_lengthout'].lower() in ['', 'none']:
                self.batch_lengthout = None
            else:
                self.batch_lengthout = int(config['software']['batch_lengthout'])

        # reset everything
        self.reset_timestamp()
        self.reset_logger()
        self.reset_workdir()
        self.reset_contents()

    def reset_contents(self):
        if not self.init_performed:
            self.init_performed = True
            if self.init_cmd:
                with contextlib.chdir(self.path):
                    timeout = self.init_timeout or magpie.settings.default_timeout
                    lengthout = self.init_lengthout or magpie.settings.default_lengthout
                    exec_result = self.exec_cmd(shlex.split(self.init_cmd),
                                                timeout=timeout,
                                                lengthout=lengthout)
                    run_result = RunResult(None, exec_result.status)
                    if run_result.status == 'SUCCESS':
                        self.process_init_exec(run_result, exec_result)
                    if run_result.status != 'SUCCESS':
                        run_result.status = f'INIT_{run_result.status}'
                        run_result.last_exec = exec_result
                        self.diagnose_error(run_result)
                        msg = '(cmd_init) failed to init target software'
                        raise RuntimeError(msg)
        super().reset_contents()

    def evaluate_variant(self, variant, cached_run=None):
        # check batch sync
        if cached_run is None:
            # new variant
            self.write_variant(variant)
        elif not cached_run.cache.keys():
            # cached (failed) --> early exit
            return cached_run
        elif {inst for b in self.batch for inst in b}.issubset(cached_run.cache.keys()):
            # cached (complete) --> early exit
            self.process_batch_final(cached_run)
            return cached_run
        else:
            # partially cached
            self.write_variant(variant)

        # evaluate
        work_path = self.work_dir / self.basename
        run_result = cached_run or RunResult(variant, 'UNKNOWN_ERROR')
        run_result.updated = True

        with contextlib.chdir(work_path):
            # one-time setup
            if not self.setup_performed:
                self.setup_performed = True

                # make sure this is the unmodified software
                for filename in self.target_files:
                    model = variant.models[filename]
                    if model.dump() != model.cached_dump:
                        raise AssertionError

                # run "[software] setup_cmd" if provided
                if self.setup_cmd:
                    # setup
                    cli = self.compute_local_cli(variant, 'setup')
                    setup_cmd = self.setup_cmd.strip()
                    if '{PARAMS}' in self.setup_cmd:
                        setup_cmd = setup_cmd.replace('{PARAMS}', cli)
                    else:
                        setup_cmd = f'{setup_cmd} {cli}'
                    timeout = self.setup_timeout or magpie.settings.default_timeout
                    lengthout = self.setup_lengthout or magpie.settings.default_lengthout
                    exec_result = self.exec_cmd(shlex.split(setup_cmd),
                                                timeout=timeout,
                                                lengthout=lengthout)
                    
                    run_result.status = exec_result.status
                    run_result.last_exec = exec_result
                    if run_result.status == 'SUCCESS':
                        self.process_setup_exec(run_result, exec_result)
                    if run_result.status != 'SUCCESS':
                        run_result.status = f'SETUP_{run_result.status}'
                        return run_result

                # sync work directory
                self.sync_folder(self.path, work_path)

            # run "[software] compile_cmd" if provided
            if self.compile_cmd:
                cli = self.compute_local_cli(variant, 'compile')
                compile_cmd = self.compile_cmd.strip()
                if '{PARAMS}' in self.compile_cmd:
                    compile_cmd = compile_cmd.replace('{PARAMS}', cli)
                else:
                    compile_cmd = f'{compile_cmd} {cli}'
                timeout = self.compile_timeout or magpie.settings.default_timeout
                lengthout = self.compile_lengthout or magpie.settings.default_lengthout
                exec_result = self.exec_cmd(shlex.split(compile_cmd),
                                            timeout=timeout,
                                            lengthout=lengthout)
                run_result.status = exec_result.status
                run_result.last_exec = exec_result
                if run_result.status == 'SUCCESS':
                    self.process_compile_exec(run_result, exec_result)
                if run_result.status != 'SUCCESS':
                    run_result.status = f'COMPILE_{run_result.status}'
                    return run_result

            # run "[software] test_cmd" if provided
            if self.test_cmd:
                cli = self.compute_local_cli(variant, 'test')
                test_cmd = self.test_cmd.strip()
                if '{PARAMS}' in self.test_cmd:
                    test_cmd = test_cmd.replace('{PARAMS}', cli)
                else:
                    test_cmd = f'{test_cmd} {cli}'
                timeout = self.test_timeout or magpie.settings.default_timeout
                lengthout = self.test_lengthout or magpie.settings.default_lengthout
                exec_result = self.exec_cmd(shlex.split(test_cmd),
                                            timeout=timeout,
                                            lengthout=lengthout)
                # exec_result=self.exec_cmd_record(test_cmd+" "+cli,  
                #                                 timeout=timeout,
                #                                 lengthout=lengthout,
                #                                 retries=1)
                
                
                run_result.status = exec_result.status
                run_result.last_exec = exec_result
                if run_result.status == 'SUCCESS':
                    self.process_test_exec(run_result, exec_result)
                if run_result.status != 'SUCCESS':
                    run_result.status = f'TEST_{run_result.status}'
                    return run_result

            # when fitness is computed from test_cmd, run_cmd is irrelevant
            if self.fitness_type in ['repair', 'bloat_lines', 'bloat_words', 'bloat_chars']:
                return run_result

            # run "[software] run_cmd" if provided
            if self.run_cmd:
                cli = self.compute_local_cli(variant, 'run')
                timeout = self.run_timeout or magpie.settings.default_timeout
                lengthout = self.run_lengthout or magpie.settings.default_lengthout
                retries = self.retries or 1
                print(f"Retries: {retries}")
                batch_timeout = self.batch_timeout
                batch_lengthout = self.batch_lengthout
                insts = [inst for b in self.batch for inst in b]
                for inst in insts:
                    if inst in run_result.cache:
                        continue
                    run_cmd = self.run_cmd.strip()
                    if '{INST}' in self.run_cmd:
                        
                        run_cmd = run_cmd.replace('{INST}', inst)
                    else:
                        run_cmd = f'{run_cmd} {inst}'
                    if '{PARAMS}' in self.run_cmd:
                        # print(f"Run command: {self.run_cmd}")
                        run_cmd = run_cmd.replace('{PARAMS}', cli)
                        # print(f"Run command: {run_cmd}")

                    else:
                        run_cmd = f'{run_cmd} {cli}'
                    #print(f"Running command: {run_cmd}")
                    #shlex.split(run_cmd)
                    #self.run_command(run_cmd+" "+cli)
                    #if run_cmd contains "./run_custom.sh"
                    # print(f"Run command: {run_cmd}")
                    # print(f"CLI: {cli}")
                    if run_cmd.find("./run_custom.sh") != -1:
                         exec_result = self.exec_cmd_record(run_cmd+" "+cli,  
                                                timeout=timeout,
                                                lengthout=lengthout,
                                                retries=retries)
                    else:
                        exec_result = self.exec_cmd_retries(shlex.split(run_cmd),  
                                                    timeout=timeout,
                                                    lengthout=lengthout,
                                                    retries=retries)
                    run_result.status = exec_result.status
                    run_result.last_exec = exec_result
                    if run_result.status == 'SUCCESS':
                        self.process_run_exec(run_result, exec_result)
                    self.process_batch_single(run_result, inst)
                    if run_result.status != 'SUCCESS':
                        run_result.status = f'RUN_{run_result.status}'
                        break
                    if batch_timeout:
                        batch_timeout -= exec_result.runtime
                        if batch_timeout < 0:
                            run_result.status = 'BATCH_TIMEOUT'
                            break
                    if batch_lengthout:
                        batch_lengthout -= exec_result.output_length
                        if batch_lengthout < 0:
                            run_result.status = 'BATCH_LENGTHOUT'
                            break

        # final process
        self.process_batch_final(run_result)
        return run_result

    def compute_local_cli(self, variant, step):
        cli = ''
        for target in self.target_files:
            model = variant.models[target]
            cli = model.update_cli(variant, cli, step)
        return cli

    def process_init_exec(self, run_result, exec_result):
        # "[software] init_cmd" must yield nonzero return code
        if exec_result.return_code != 0:
            run_result.status = 'CODE_ERROR'

    def process_setup_exec(self, run_result, exec_result):
        # "[software] setup_cmd" must yield nonzero return code
        if exec_result.return_code != 0:
            run_result.status = 'CODE_ERROR'

    def process_compile_exec(self, run_result, exec_result):
        # "[software] compile_cmd" must yield nonzero return code
        if exec_result.return_code != 0:
            run_result.status = 'CODE_ERROR'

    def process_test_exec(self, run_result, exec_result):
        # if "[software] fitness" is "repair", we check STDOUT for the number of failed test cases
        if self.fitness_type == 'repair':
            stdout = exec_result.stdout.decode(magpie.settings.output_encoding)
            for fail_regexp, total_regexp in [
                (r'Failures: (\d+)\b', r'^Tests run: (\d+)\b'), # junit
                (r'\b(\d+) (?:failed|error)', r'^collected (\d+) items'), # pytest
                (r' (\d+) (?:failures|errors)', r'^(\d+) runs,'), # minitest
            ]:
                fail_matches = re.findall(fail_regexp, stdout, re.MULTILINE)
                total_matches = re.findall(total_regexp, stdout, re.MULTILINE)
                n_fail = 0
                n_total = 0
                try:
                    for m in fail_matches:
                        n_fail += float(m)
                    for m in total_matches:
                        n_total += float(m)
                except ValueError:
                    run_result.status = 'PARSE_ERROR'
                if n_total > 0:
                    run_result.fitness = round(100*n_fail/n_total, 2)
                    break
            else:
                run_result.status = 'PARSE_ERROR'

        # in all other cases "[software] test_cmd" must just yield nonzero return code
        elif exec_result.return_code != 0:
            run_result.status = 'CODE_ERROR'
            return

        # if "[software] fitness" is one of "bloat_*", we can count here
        if self.fitness_type[:6] == 'bloat_':
            run_result.fitness = 0
            for filename in self.target_files:
                renamed = run_result.variant.models[filename].renamed_filename
                with pathlib.Path(renamed).open('r') as target:
                    if self.fitness_type == 'bloat_lines':
                        run_result.fitness += len(target.readlines())
                    elif self.fitness_type == 'bloat_words':
                        run_result.fitness += sum(len(s.split()) for s in target.readlines())
                    elif self.fitness_type == 'bloat_chars':
                        run_result.fitness += sum(len(s) for s in target.readlines())
                    

    def process_run_exec(self, run_result, exec_result):
        # in all cases "[software] run_cmd" must yield nonzero return code
        if exec_result.return_code != 0:
            run_result.status = 'CODE_ERROR'
            return

        # if "[software] fitness" is "output", we check STDOUT for the string "MAGPIE_FITNESS:"
        if self.fitness_type == 'output':
            stdout = exec_result.stdout.decode(magpie.settings.output_encoding)
            m = re.findall('MAGPIE_FITNESS: (.*)', stdout)
            try:
                fitnesses = [float(time.strip()) for time in m]
                run_result.fitness = sum(fitnesses)/len(fitnesses)
                # self.logger.info(f'Fitness value === {run_result.fitness}')
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "time", we just use time as seen by the main Python process
        elif self.fitness_type == 'time':
            run_result.fitness = round(exec_result.runtime, 4)
            # self.logger.info(f'Fitness value === {run_result.fitness}')

        # if "[software] fitness" is "posix_time", we assume a POSIX-compatible output on STDERR
        elif self.fitness_type == 'posix_time':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            m = re.findall('real (.*)', stderr)
            try:
                ptimes =[float(time.strip()) for time in m]
                run_result.fitness = sum(ptimes)/len(ptimes)
                # self.logger.info(f'Fitness value === {run_result.fitness}')
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_time", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_time':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m =  re.findall(r'\s*([0-9.,]+)\s+seconds time elapsed\s*', stderr)
            if m:
                elapsed_times = [round(float(time.strip().replace(',', '.')),8) for time in m]
                #print(f"Elapsed times: -{elapsed_times}- seconds")
                # Calculate the average if more than one time is found
                elapsed_time = sum(elapsed_times) / len(elapsed_times)
                print(f"Elapsed time: -{elapsed_time}- seconds")
            else:
                print("Elapsed time not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = elapsed_time
                # self.logger.info(f'Fitness value === {run_result.fitness}')
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'
        
        # if "[software] fitness" is "perf_cycles", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_cycles':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+cycles\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                cycles = sum(temp) / len(temp)
                print(f"Cycles: -{cycles}- cycles")
            else:
                print("Cycles not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = cycles
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_instructions", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_instructions':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+instructions\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                insts = sum(temp) / len(temp)
                print(f"Perf instructions: -{insts}- instructions")
            else:
                print("Perf Instructions not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = insts
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'
        
        # if "[software] fitness" is "perf_cache_references", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_cache_references':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+cache-references\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                cache_refs = sum(temp) / len(temp)
                print(f"cache-references: -{cache_refs}")
            else:
                print("Cache-references not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = cache_refs
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_cache_misses", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_cache_misses':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+cache-misses\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                misses = sum(temp) / len(temp)
                print(f"Misses: -{misses}-")
            else:
                print("Perf cache misses not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = misses
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_branches", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_branches':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+branches\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                branches = sum(temp) / len(temp)
                print(f"Branches: -{branches}- branches")
            else:
                print("Branches not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = branches
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_branch_misses", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_branch_misses':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+branch-misses\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                branch_misses = sum(temp) / len(temp)
                print(f"branch-misses: -{branch_misses}- branch-misses")
            else:
                print("Branch misses not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = branch_misses
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'
        
        # if "[software] fitness" is "perf_cpu_clock", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_cpu_clock':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+msec cpu-clock\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),4) for val in m]
                clocks = sum(temp) / len(temp)
                print(f"clocks: -{clocks}- msec")
            else:
                print("Cpu clocks not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = clocks
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'
    
        # if "[software] fitness" is "perf_task_clock", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_task_clock':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+msec task-clock\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),4) for val in m]
                task_clock = sum(temp) / len(temp)
                print(f"task_clock: -{task_clock}- msec")
            else:
                print("task_clock not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = task_clock
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'
        
        # if "[software] fitness" is "perf_faults", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_faults':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+faults\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                faults = sum(temp) / len(temp)
                print(f"faults: -{faults}- ")
            else:
                print("faults not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = faults
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_major_faults", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_major_faults':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+major-faults\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                maj_faults = sum(temp) / len(temp)
                print(f"major faults: -{maj_faults}-")
            else:
                print("major_faults not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = maj_faults
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_minor_faults", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_minor_faults':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+minor-faults\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                min_faults = sum(temp) / len(temp)
                print(f"minor faults: -{min_faults}-")
            else:
                print("min_faults not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = min_faults
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_cs", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_cs':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+cs\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                cs = sum(temp) / len(temp)
                print(f"contex switchs: -{cs}- ")
            else:
                print("contex switches not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = cs
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_migrations", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_migrations':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+migrations\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                migrations = sum(temp) / len(temp)
                print(f"migrations: -{migrations}- ")
            else:
                print("migrations not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = migrations
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'
        
        # if "[software] fitness" is "perf_L1_dcache_loads", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_L1_dcache_loads':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+L1-dcache-loads\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                l1_loads = sum(temp) / len(temp)
                print(f"l1_loads: -{l1_loads}- ")
            else:
                print("l1_loads not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = l1_loads
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_L1_dcache_load_misses", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_L1_dcache_load_misses':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+L1-dcache-load-misses\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                L1_misses = sum(temp) / len(temp)
                print(f"L1_misses: -{L1_misses}-")
            else:
                print("L1_misses not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = L1_misses
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_dTLB_loads", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_dTLB_loads':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+dTLB-loads\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                dtlb_loads = sum(temp) / len(temp)
                print(f"dtlb_loads: -{dtlb_loads}-")
            else:
                print("dtlb_loads not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = dtlb_loads
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_dTLB_load_misses", we assume a perf-like output on STDERR
        elif self.fitness_type == 'perf_dTLB_load_misses':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            m = re.findall(r'\s*([0-9,]+)\s+dTLB-load-misses\s*', stderr)
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                dtlb_misses = sum(temp) / len(temp)
                print(f"dtlb_misses: -{dtlb_misses}-")
            else:
                print("dtlb_misses not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = dtlb_misses
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "weights", we assume our custom  output on STDERR
        elif self.fitness_type == 'weights':
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            pattern = r"Custom Metric: ([\d\.]+)"

            # Use re.findall to find the pattern in the given line
            m = re.findall(pattern, stderr)
            
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                # print(f"All weights: -{temp}-")
                custom = sum(temp) / len(temp)
                print(f"Custom weights: -{custom}-")
            else:
                print("Custom weight metric not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = custom
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

       # if "[software] fitness" is "energy", we assume our custom  output on STDERR
        elif self.fitness_type == 'energy':
            
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            pattern = r"Energy consumed: (\d+)"

            # Use re.findall to find the pattern in the given line
            m = re.findall(pattern, stderr)
            print(f"STDERR: {stderr}")
            
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                energy = sum(temp) / len(temp)
                print(f"Rapl energy: -{energy}-")
            else:
                print("energy rapl metric not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = energy
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "energy_ram", we assume our custom  output on STDERR
        elif self.fitness_type == 'energy_ram':
            
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            pattern = r"Energy ram consumed: (\d+)"

            # Use re.findall to find the pattern in the given line
            m = re.findall(pattern, stderr)
            print(f"STDERR: {stderr}")
            
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                energy_ram = sum(temp) / len(temp)
                print(f"Rapl energy_ram: -{energy_ram}-")
            else:
                print("energy_ram rapl metric not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = energy_ram
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'
        
        # if "[software] fitness" is "energy_uncore", we assume our custom  output on STDERR
        elif self.fitness_type == 'energy_uncore':
            
            stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
            #print(stderr)
            #m = re.findall('(.*) seconds time elapsed', stderr)
            pattern = r"Energy uncore consumed: (\d+)"

            # Use re.findall to find the pattern in the given line
            m = re.findall(pattern, stderr)
            print(f"STDERR: {stderr}")
            
            if m:
                temp = [round(float(val.strip().replace(',', '.')),3) for val in m]
                energy_uncore = sum(temp) / len(temp)
                print(f"Rapl energy_uncore: -{energy_uncore}-")
            else:
                print("energy_uncore rapl metric not found")
                run_result.status = 'PARSE_ERROR'
            try:
                run_result.fitness = energy_uncore
                # self.logger.info(f'Fitness value === {run_result.fitness}')
                print(f"Fitness: {run_result.fitness}")
            except (AttributeError, ValueError):
                run_result.status = 'PARSE_ERROR'

        # if "[software] fitness" is "perf_instructions", we assume a perf-like output on STDERR
        # elif self.fitness_type == 'perf_instructions':
        #     stderr = exec_result.stderr.decode(magpie.settings.output_encoding)
        #     m = re.search('(.*) instructions', stderr)
        #     try:
        #         run_result.fitness = int(m.group(1).replace(',', ''))
        #     except (AttributeError, ValueError):
        #         run_result.status = 'PARSE_ERROR'

    def process_batch_single(self, run_result, inst):
        run_result.cache[inst] = (run_result.status, run_result.fitness)
        self.logger.debug('EXEC> %s %s %s', inst, run_result.status, run_result.fitness)

    def process_batch_final(self, run_result):
        tmp = []
        fit = []
        for b in self.batch:
            bin_fitness = []
            for inst in b:
                status, fitness = run_result.cache[inst]
                if status != 'SUCCESS':
                    # TODO: penalised fitness
                    run_result.fitness = None
                    return
                bin_fitness.append(fitness)
            multi = isinstance(bin_fitness[0], list)
            if self.batch_bin_fitness_strategy == 'aggregate':
                tmp.extend(bin_fitness)
            else:
                for a in ([list(a) for a in zip(*bin_fitness)] if multi else [bin_fitness]):
                    precision = max(len(str(float(x)).split('.')[1]) for x in a)
                    if self.batch_bin_fitness_strategy == 'sum':
                        v = sum(a)
                    elif self.batch_bin_fitness_strategy == 'average':
                        v = sum(a)/len(a)
                        precision += 1
                    elif self.batch_bin_fitness_strategy == 'median':
                        if len(a) % 2 == 0:
                            v = sorted(a)[len(a)//2]
                        else:
                            v = sum(sorted(a)[len(a)//2:len(a)//2+2])/2
                            precision += 1
                    elif self.batch_bin_fitness_strategy == 'q10':
                        if len(a) % 10 == 0:
                            v = sorted(a)[len(a)//10]
                        else:
                            v = sum(sorted(a)[len(a)//10:len(a)//10+2])/2
                            precision += 1
                    elif self.batch_bin_fitness_strategy == 'q25':
                        if len(a) % 4 == 0:
                            v = sorted(a)[len(a)//4]
                        else:
                            v = sum(sorted(a)[len(a)//10:len(a)//10+2])/2
                            precision += 1
                    elif self.batch_bin_fitness_strategy == 'q75':
                        if len(a) % 4 == 0:
                            v = sorted(a)[3*len(a)//4]
                        else:
                            v = sum(sorted(a)[3*len(a)//4:3*len(a)//4+2])/2
                            precision += 1
                    elif self.batch_bin_fitness_strategy == 'q90':
                        if len(a) % 10 == 0:
                            v = sorted(a)[len(a)//10]
                        else:
                            v = sum(sorted(a)[9*len(a)//10:9*len(a)//10+2])/2
                            precision += 1
                    tmp.append(round(v, precision))
        for a in ([list(a) for a in zip(*tmp)] if multi else [tmp]):
            precision = max(len(str(float(x)).split('.')[1]) for x in a)
            if self.batch_fitness_strategy == 'sum':
                v = sum(a)
            elif self.batch_fitness_strategy == 'average':
                v = sum(a)/len(a)
                precision += 1
            elif self.batch_fitness_strategy == 'median':
                if len(a) % 2 == 0:
                    v = sorted(a)[len(a)//2]
                else:
                    v = sum(sorted(a)[len(a)//2:len(a)//2+2])/2
                    precision += 1
            fit.append(round(v, precision))
        run_result.fitness = fit if multi else fit[0]

    def diagnose_error(self, run):
        self.logger.info('!*'*40)
        self.logger.info('Unable to run and evaluate the target software.')
        self.logger.info('Self-diagnostic:')
        self.self_diagnostic(run)
        self.logger.info('!*'*40)
        if run.last_exec is not None:
            self.logger.info('CWD: %s', pathlib.Path(self.work_dir) / self.basename)
            self.logger.info('CMD: %s', run.last_exec.cmd)
            self.logger.info('STATUS: %s', run.last_exec.status)
            self.logger.info('RETURN_CODE: %d', run.last_exec.return_code)
            self.logger.info('RUNTIME: %s', run.last_exec.runtime)
            encoding = magpie.settings.output_encoding
            self.logger.info('STDOUT: (see log file)')
            try:
                s = run.last_exec.stdout.decode(encoding)
                self.logger.debug('STDOUT:\n%s', s)
            except UnicodeDecodeError:
                self.logger.debug('STDOUT: (failed to decode to: %s)\n%s', encoding, run.last_exec.stdout)
            self.logger.info('STDERR: (see log file)')
            try:
                s = run.last_exec.stderr.decode(encoding)
                self.logger.debug('STDERR:\n%s', s)
            except UnicodeDecodeError:
                self.logger.debug('STDERR: (failed to decode to: %s)\n%s', encoding, run.last_exec.stderr)
            self.logger.info('!*'*40)

    def self_diagnostic(self, run):
        for step in ['init', 'setup', 'compile', 'test', 'run']:
            if run.status == f'{step.upper()}_CLI_ERROR':
                self.logger.info('Unable to run the "%s_cmd" command', step)
                self.logger.info('--> there might be a typo (try it manually)')
                self.logger.info('--> your command might not be found (check your PATH)')
                self.logger.info('--> it might not run from the correct directory (check CWD below)')
            if run.status == f'{step.upper()}_CODE_ERROR':
                self.logger.info('The "%s_cmd" command failed with a nonzero exit code', step)
                self.logger.info('--> try to run it manually')
            if run.status == f'{step.upper()}_PARSE_ERROR':
                self.logger.info('The "%s_cmd" STDOUT/STDERR was invalid', step)
                self.logger.info('--> try to run it manually')
            if run.status == f'{step.upper()}_TIMEOUT':
                self.logger.info('The "%s_cmd" command took too long to run', step)
                self.logger.info('--> consider increasing "%s_timeout"', step)
            if run.status == f'{step.upper()}_LENGTHOUT':
                self.logger.info('The "%s_cmd" command generated too much output', step)
                self.logger.info('--> consider increasing "%s_lengthout"', step)
        if run.status == 'BATCH_TIMEOUT':
            self.logger.info('Batch execution of "run_cmd" took too long to run')
            self.logger.info('--> consider increasing "batch_timeout"')
        if run.status == 'BATCH_LENGTHOUT':
            self.logger.info('Batch execution of "run_cmd" generated too much output')
            self.logger.info('--> consider increasing "batch_lengthout"')

magpie.utils.known_software.append(BasicSoftware)
