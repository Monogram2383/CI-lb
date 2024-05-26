import random
import time
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pprint import pprint

random.seed(1)

@dataclass
class Item:
    name: str
    price: int
    weight: int


class GeneticAlgorithm:
    def __init__(self, genome_length: int, population_size: int, mutation_probability: float = 0.2,
                 max_mutated_genes: int = 1):
        self.genome_length: int = genome_length
        self.population_size: int = population_size

        self.mutation_probability: float = mutation_probability
        self.max_mutated_genes: int = max_mutated_genes



    def generate_genome(self):
        '''
        Cтворює геном у вигляді одновимірного масиву даних довжиною genome_length з бінарними значеннями
        :param genome_length: довжина геному
        :return: бінарний масив, що інтерпретує геномa
        '''
        return random.choices([0, 1], k=self.genome_length)

    def generate_population(self) -> list:
        return [self.generate_genome() for _ in range(self.population_size)]


    def selection(self, population: list, items, weight_limit, new_population_size: int = 2) -> list:
        return random.choices(
            population=population,
            weights=[self.fitness(genome, items, weight_limit) for genome in population],
            k=new_population_size
        )

    def crossover(self, parent_a: list, parent_b: list) -> tuple[list, list]:
        '''
        Функція імплементації "одноточкового кросовера". Бере геном двох батьків та поєднує їх для отримання двох нових
        геномів з розділенням у випадковій точці (індексі) `p`
        :param parent_a: перший з батьків
        :param parent_b: другий з батьків
        :return: два нових генома
        '''
        p = random.randint(0, self.genome_length - 1) # індекс розділення геномів
        return parent_a[0: p] + parent_b[p:], parent_b[0:p] + parent_a[p:]

    def mutation(self, genome):
        mutated_genes = 0
        for i, gene in enumerate(genome):
            if mutated_genes >= self.max_mutated_genes:
                break
            if random.random() >= self.mutation_probability:
                genome[i] = abs(gene - 1) # abs(1 - 1) = 0, abs(0 - 1) = 1
                mutated_genes += 1
        return genome

    def fitness(self, genome: list, items: list[Item], weight_limit: int):

        weight = sum([item.weight for i, item in enumerate(items) if genome[i] == 1])
        price = sum([item.price for i, item in enumerate(items) if genome[i] == 1])

        if weight > weight_limit:
            return 0
        return price


    def run(self, items: list[Item], fitness_limit: int, weight_limit: int,
            max_generations: int = 1000, leave_top_n: int = 2, allow_elitism: bool = False):
        population: list = self.generate_population()
        metrics = {
            "top_fitness": [],
            "avg_fitness": [],
        }
        for gen_i in range(max_generations):
            population = self.evaluate_population(population, items, weight_limit)

            top_fitness: int = self.fitness(population[0], items, weight_limit)
            avg_fitness: float = np.mean([self.fitness(genome, items, weight_limit)
                                  for genome in population])
            if top_fitness > fitness_limit:
                print(f"| Generation #{gen_i}: top_fitness = {top_fitness}, avg_fitness = {avg_fitness} |")
                metrics["top_fitness"].append(top_fitness)
                metrics["avg_fitness"].append(avg_fitness)
                break

            # ініціація наступного покоління + елітизм (обрати leave_top_n геномів у наступне покоління)
            if allow_elitism:
                next_generation = population[:leave_top_n]
                parent_couples = int(self.population_size / 2 - 1)
            else:
                next_generation = []
                parent_couples = int(self.population_size / 2)

            # додати нові геноми у наступне покоління
            for j in range(parent_couples):
                # отримати батьків (selection)
                parent_a, parent_b = self.selection(population,items, weight_limit, 2)
                # створити дітей (crossover)
                child_a, child_b = self.crossover(parent_a, parent_b)
                # мутація дітей (mutation)
                child_a, child_b = self.mutation(child_a), self.mutation(child_b)
                # додати дітей до популяції
                next_generation += [child_a, child_b]
            population = next_generation

            # print(f"| Generation #{gen_i}: top_fitness = {top_fitness}, avg_fitness = {avg_fitness} |")
            metrics["top_fitness"].append(top_fitness)
            metrics["avg_fitness"].append(avg_fitness)
        population = self.evaluate_population(population, items, weight_limit)
        return population, gen_i, metrics

    def evaluate_population(self, population: list, items: list[Item], weight_limit: int) -> list:
        return sorted(
            population,
            key=lambda genome: self.fitness(genome, items, weight_limit),
            reverse=True
        )

def brute_force(items: list[Item], weight_limit: int) -> int:
    """
    Спроба знайти максимальне значення через перебір всіх можливих комбінацій. Функція для порівняння необхідного часу
    :param items: перелік опцій з яких можна обрати речі
    :param weight_limit: обмеження у вазі рюкзака
    :return: максимальний фітнес
    """
    if len(items) == 0:
        return 0

    max_price = 0
    for i, item in enumerate(items):
        if item.weight > weight_limit:
            continue
        price = brute_force(items[i + 1:], weight_limit - item.weight)
        if price + item.price >= max_price:
            max_price = price + item.price
    return max_price

if __name__ == '__main__':
    """
    - ми вирішуємо Knapsack Problem. Створюємо N випадкових згенерованих Item з вагою та цінністю
    - ставимо 4 експерименти з різною кількістю Item: [10, 25, 50, 100]
    - у всіх встановлюємо ліміти на вагу та фітнес (60% та 80% від загальної кількості)
    - популяція всюди дорівнює 10
    - оптимізуємо вартість/цінність ранця
    - кількість максимальних поколінь дорівнює 1000
    - візуалізуємо
    """
    for allow_elitism in [True, False]:
        for N in [10, 25, 50, 100]:
            knapsack = [
                Item(name=f"Item #{i}", weight=random.randint(10, 150) * 10, price=random.randint(1, 30) * 10)
                for i in range(N)
            ]
            print("===============================================")
            print("Knapsack:")
            pprint(knapsack)
            print("===============================================")
            fitness_limit = sum([item.price for item in knapsack]) * 0.8
            weight_limit = sum([item.weight for item in knapsack]) * 0.7
            print(f"Fitness limit: {fitness_limit}")
            print(f"Weight limit: {weight_limit}")
            print(f"Allow elitism: {allow_elitism}")
            print("===============================================")

            GA = GeneticAlgorithm(genome_length=N, population_size=10, mutation_probability=0.2, max_mutated_genes=1)
            population, gen_i, metrics = GA.run(knapsack, fitness_limit=fitness_limit, weight_limit=weight_limit, allow_elitism=allow_elitism)


            print(f"Needed generations: {gen_i}; final fitness: {metrics['top_fitness'][-1]}; fitness limit: {fitness_limit}")
            x_axis_len = len(metrics['top_fitness'])
            plt.plot(range(x_axis_len), metrics['top_fitness'], label='Top fitness')
            plt.plot(range(x_axis_len), metrics['avg_fitness'], label='Average fitness')
            plt.plot(range(x_axis_len), [fitness_limit] * x_axis_len, label='Fitness limit')
            plt.xlabel('Generations')
            plt.ylabel('Fitness')
            plt.title(f"GA fitness history (N={N}, elitism={allow_elitism})")
            plt.legend()
            plt.show()

    # Порівняння часу з брут форсом
    time_needed_ga = [] # кількість часу для вирішення задачі генетичним алгоритмом
    time_needed_bf = [] # кількість часу для вирішення задачі перебором всіх комбінацій
    for N in range(10, 30):
        knapsack = [
            Item(name=f"Item #{i}", weight=random.randint(10, 150) * 10, price=random.randint(1, 30) * 10)
            for i in range(N)
        ]
        fitness_limit = sum([item.price for item in knapsack]) * 0.8
        weight_limit = sum([item.weight for item in knapsack]) * 0.7

        GA = GeneticAlgorithm(genome_length=N, population_size=10, mutation_probability=0.2, max_mutated_genes=1)
        start = time.time()
        population, gen_i, metrics = GA.run(knapsack, fitness_limit=fitness_limit, weight_limit=weight_limit,
                                            allow_elitism=allow_elitism)
        end = time.time()
        time_needed_ga.append(end - start)

        # brute_force result
        start = time.time()
        brute_force_result = brute_force(items=knapsack, weight_limit=weight_limit)
        end = time.time()
        time_needed_bf.append(end - start)

        print(f"N={N}; brute force time: {time_needed_bf[-1]}; GA time: {time_needed_ga[-1]}")
    x_axis = range(10, 30)
    plt.plot(x_axis, time_needed_ga, label='Time needed GA')
    plt.plot(x_axis, time_needed_bf, label='Time needed Brute Force')
    plt.xlabel('Length of knapsack (N)')
    plt.ylabel('Time needed (s)')
    plt.title(f"Time needed comparison elitism={allow_elitism}")
    plt.legend()
    plt.show()