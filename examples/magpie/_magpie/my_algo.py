import magpie


class MyAlgo(magpie.algos.FirstImprovement):
    def mutate(self, patch):
        for _ in range(3):
            super().mutate(patch)

magpie.utils.known_algos += [MyAlgo]
