import abc
import copy
import random

import magpie.core
import magpie.utils


class LocalSearch(magpie.core.BasicAlgorithm):
    def __init__(self):
        super().__init__()
        self.name = 'Local Search'
        self.config['delete_prob'] = 0.5
        self.config['max_neighbours'] = None
        self.config['when_trapped'] = 'continue'

    def reset(self):
        super().reset()
        self.stats['neighbours'] = 0

    def setup(self, config):
        super().setup(config)
        sec = config['search.ls']
        self.config['delete_prob'] = float(sec['delete_prob'])
        self.config['max_neighbours'] = int(val) if (val := sec['max_neighbours']) else None
        self.config['when_trapped'] = sec['when_trapped']

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

            # main loop
            current_patch = self.report['best_patch']
            current_fitness = self.report['best_fitness']
            while not self.stopping_condition():
                self.hook_main_loop()
                current_patch, current_fitness = self.explore(current_patch, current_fitness)

        except KeyboardInterrupt:
            self.report['stop'] = 'keyboard interrupt'

        finally:
            # the end
            self.hook_end()

    @abc.abstractmethod
    def explore(self, current_patch, current_fitness):
        pass

    def mutate(self, patch):
        n = len(patch.edits)
        if n == 0:
            if self.config['delete_prob'] == 1:
                self.report['stop'] = 'trapped'
            else:
                patch.edits.append(self.create_edit(self.software.noop_variant))
        elif random.random() < self.config['delete_prob']:
            del patch.edits[random.randrange(0, n)]
        else:
            patch.edits.append(self.create_edit(self.software.noop_variant))

    def check_if_trapped(self):
        if self.config['max_neighbours'] is None:
            return
        if self.stats['neighbours'] < self.config['max_neighbours']:
            return
        if self.config['when_trapped'] == 'stop':
            self.report['stop'] = 'trapped'
        # TODO: restart, others?


class DummySearch(LocalSearch):
    def __init__(self):
        super().__init__()
        self.name = 'Dummy Search'

    def explore(self, current_patch, current_fitness):
        self.report['stop'] = 'dummy end'
        return current_patch, current_fitness

magpie.utils.known_algos.append(DummySearch)


class DebugSearch(LocalSearch):
    def __init__(self):
        super().__init__()
        self.name = 'Debug Search'

    def explore(self, current_patch, current_fitness):
        debug_patch = self.report['debug_patch']
        for edit in debug_patch.edits:
            # move
            patch = magpie.core.Patch([edit])

            # compare
            variant = magpie.core.Variant(self.software, patch)
            run = self.evaluate_variant(variant)
            accept = best = False
            if run.status == 'SUCCESS':
                accept = True
                if self.dominates(run.fitness, self.report['best_fitness']):
                    self.report['best_fitness'] = run.fitness
                    self.report['best_patch'] = patch
                    best = True

            # hook
            self.hook_evaluation(patch, run, accept, best)

            # next
            self.stats['steps'] += 1
        self.report['stop'] = 'debug end'
        return current_patch, current_fitness

magpie.utils.known_algos.append(DebugSearch)


class RandomSearch(LocalSearch):
    def __init__(self):
        super().__init__()
        self.name = 'Random Search'

    def explore(self, current_patch, current_fitness):
        # move
        patch = magpie.core.Patch()
        self.mutate(patch)

        # compare
        variant = magpie.core.Variant(self.software, patch)
        run = self.evaluate_variant(variant)
        best = False
        if run.status == 'SUCCESS':
            if self.dominates(run.fitness, self.report['best_fitness']):
                self.report['best_fitness'] = run.fitness
                self.report['best_patch'] = patch
                best = True

        # hook
        self.hook_evaluation(variant, run, False, best)

        # next
        self.stats['steps'] += 1
        return patch, run.fitness

magpie.utils.known_algos.append(RandomSearch)


class RandomWalk(LocalSearch):
    def __init__(self):
        super().__init__()
        self.name = 'Random Walk'
        self.config['accept_fail'] = False

    def setup(self, config):
        super().setup(config)
        sec = config['search.ls']
        self.config['accept_fail'] = sec['accept_fail']

    def explore(self, current_patch, current_fitness):
        # move
        patch = copy.deepcopy(current_patch)
        self.mutate(patch)

        # compare
        variant = magpie.core.Variant(self.software, patch)
        run = self.evaluate_variant(variant)
        accept = self.config['accept_fail']
        best = False
        if run.status == 'SUCCESS':
            accept = True
            if self.dominates(run.fitness, self.report['best_fitness']):
                self.report['best_fitness'] = run.fitness
                self.report['best_patch'] = patch
                best = True

        # accept
        if accept:
            current_patch = patch
            current_fitness = run.fitness
            self.stats['neighbours'] = 0
        else:
            self.stats['neighbours'] += 1
            self.check_if_trapped()

        # hook
        self.hook_evaluation(patch, run, accept, best)

        # next
        self.stats['steps'] += 1
        return current_patch, current_fitness

magpie.utils.known_algos.append(RandomWalk)


class FirstImprovement(LocalSearch):
    def __init__(self):
        super().__init__()
        self.name = 'First Improvement'
        self.local_tabu = set()

    def explore(self, current_patch, current_fitness):
        # move
        while True:
            patch = copy.deepcopy(current_patch)
            self.mutate(patch)
            if patch not in self.local_tabu:
                break

        # compare
        variant = magpie.core.Variant(self.software, patch)
        run = self.evaluate_variant(variant)
        accept = best = False
        if run.status == 'SUCCESS':
            if not self.dominates(current_fitness, run.fitness):
                accept = True
                if self.dominates(run.fitness, self.report['best_fitness']):
                    self.report['best_fitness'] = run.fitness
                    self.report['best_patch'] = patch
                    best = True

        # accept
        if accept:
            current_patch = patch
            current_fitness = run.fitness
            self.local_tabu.clear()
            self.stats['neighbours'] = 0
        else:
            if len(patch.edits) < len(current_patch.edits):
                self.local_tabu.add(patch)
            self.stats['neighbours'] += 1
            self.check_if_trapped()

        # hook
        self.hook_evaluation(variant, run, accept, best)

        # next
        self.stats['steps'] += 1
        return current_patch, current_fitness

magpie.utils.known_algos.append(FirstImprovement)


class BestImprovement(LocalSearch):
    def __init__(self):
        super().__init__()
        self.name = 'Best Improvement'
        self.config['max_neighbours'] = 20
        self.local_best_patch = None
        self.local_best_fitness = None
        self.local_tabu = set()

    def explore(self, current_patch, current_fitness):
        # move
        while True:
            patch = copy.deepcopy(current_patch)
            self.mutate(patch)
            if patch not in self.local_tabu:
                break

        # compare
        variant = magpie.core.Variant(self.software, patch)
        run = self.evaluate_variant(variant)
        accept = best = False
        if run.status == 'SUCCESS':
            if not self.dominates(current_fitness, run.fitness):
                if not self.dominates(self.local_best_fitness, run.fitness):
                    self.local_best_patch = patch
                    self.local_best_fitness = run.fitness
                    if self.dominates(run.fitness, self.report['best_fitness']):
                        self.report['best_fitness'] = run.fitness
                        self.report['best_patch'] = patch
                        best = True

        # accept
        accept = self.stats['neighbours'] >= self.config['max_neighbours']
        if accept:
            if self.local_best_patch is not None:
                current_patch = self.local_best_patch
                current_fitness = self.local_best_fitness
                self.local_best_patch = None
                self.local_best_fitness = None
                self.local_tabu.clear()
                self.stats['neighbours'] = 0
            else:
                self.check_if_trapped()
        else:
            if len(patch.edits) < len(current_patch.edits):
                self.local_tabu.add(patch)
            self.stats['neighbours'] += 1
            self.check_if_trapped()

        # hook
        self.hook_evaluation(patch, run, accept, best)

        # next
        self.stats['steps'] += 1
        return current_patch, current_fitness

magpie.utils.known_algos.append(BestImprovement)


class WorstImprovement(LocalSearch):
    def __init__(self):
        super().__init__()
        self.name = 'Worst Improvement'
        self.config['max_neighbours'] = 20
        self.local_worst_patch = None
        self.local_worst_fitness = None
        self.local_tabu = set()

    def explore(self, current_patch, current_fitness):
        # move
        while True:
            patch = copy.deepcopy(current_patch)
            self.mutate(patch)
            if patch not in self.local_tabu:
                break

        # compare
        variant = magpie.core.Variant(self.software, patch)
        run = self.evaluate_variant(variant)
        accept = best = False
        if run.status == 'SUCCESS':
            if not self.dominates(current_fitness, run.fitness):
                if not self.dominates(self.local_worst_fitness, run.fitness):
                    self.local_worst_patch = patch
                    self.local_worst_fitness = run.fitness
                if self.dominates(run.fitness, self.report['best_fitness']):
                    self.report['best_fitness'] = run.fitness
                    self.report['best_patch'] = patch
                    best = True

        # accept
        accept = self.stats['neighbours'] >= self.config['max_neighbours']
        if accept:
            if self.local_worst_patch is not None:
                current_patch = self.local_worst_patch
                current_fitness = self.local_worst_fitness
                self.local_worst_patch = None
                self.local_worst_fitness = None
                self.local_tabu.clear()
                self.stats['neighbours'] = 0
            else:
                self.check_if_trapped()
        else:
            if len(patch.edits) < len(current_patch.edits):
                self.local_tabu.add(patch)
            self.stats['neighbours'] += 1
            self.check_if_trapped()

        # hook
        self.hook_evaluation(patch, run, accept, best)

        # next
        self.stats['steps'] += 1
        return current_patch, current_fitness

magpie.utils.known_algos.append(WorstImprovement)


class TabuSearch(BestImprovement):
    def __init__(self):
        super().__init__()
        self.name = 'Tabu Search'
        self.config['tabu_length'] = 10
        self.tabu_list = [magpie.core.Patch()] # queues are not iterable
        self.local_tabu = set()

    def setup(self, config):
        super().setup(config)
        sec = config['search.ls']
        self.config['tabu_length'] = sec['tabu_length']

    def explore(self, current_patch, current_fitness):
        # move
        while True:
            patch = copy.deepcopy(current_patch)
            self.mutate(patch)
            if patch not in self.tabu_list and patch not in self.local_tabu:
                break

        # compare
        variant = magpie.core.Variant(self.software, patch)
        run = self.evaluate_variant(variant)
        accept = best = False
        if run.status == 'SUCCESS':
            if not self.dominates(self.local_best_fitness, run.fitness):
                self.local_best_patch = patch
                self.local_best_fitness = run.fitness
            if self.dominates(run.fitness, self.report['best_fitness']):
                self.report['best_fitness'] = run.fitness
                self.report['best_patch'] = patch
                best = True

        # accept
        accept = self.stats['neighbours'] >= self.config['max_neighbours']
        if accept:
            if self.local_best_patch is not None:
                current_patch = self.local_best_patch
                current_fitness = self.local_best_fitness
                self.local_best_patch = None
                self.local_best_fitness = None
                self.local_tabu.clear()
                self.stats['neighbours'] = 0
                self.tabu_list.append(self.local_best_patch)
                while len(self.tabu_list) >= self.config['tabu_length']:
                    self.tabu_list.pop(0)
            else:
                self.check_if_trapped()
        else:
            if len(patch.edits) < len(current_patch.edits):
                self.local_tabu.add(patch)
            self.stats['neighbours'] += 1
            self.check_if_trapped()

        # hook
        self.hook_evaluation(patch, run, accept, best)

        # next
        self.stats['steps'] += 1
        return current_patch, current_fitness

magpie.utils.known_algos.append(TabuSearch)
