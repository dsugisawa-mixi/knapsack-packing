import os
import pickle
import time
import numpy as np
from math import ceil
from multiprocessing import Pool
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point
import evolutionary
import greedy
import reversible
import shape_functions
from circle import Circle
from common_algorithm_functions import get_time_since, visualize_boxplot_for_data_sequence, print_if_allowed, \
    visualize_plot, visualize_bar_plot, add_newlines_by_spaces, get_stats
from ellipse import Ellipse
from problem_solution import Item, Container, Problem, Solution

# types of problems that can be solved: Knapsack-Packing Joint Problem, or Packing Problem
KNAPSACK_PACKING_PROBLEM_TYPE = "KnapsackPacking"
PACKING_PROBLEM_TYPE = "Packing"

# directory where to save figures and results of the problems created specifically for the Knapsack-Packing Joint Problem
KNAPSACK_PACKING_PROBLEM_DIR = "../Output/Problems/CustomKnapsackPacking/Comparison/"

# directory where to save figures and results of instances of the Packing Problem
PACKING_PROBLEM_DIR = "../Output/Problems/Packing/Comparison/"


def create_knapsack_packing_problems_with_manual_solutions(can_print=False):

    """Create a set of Knapsack-Packing problem instances that are solved (optimally) with manual placements (using actions available for all the algorithms); both the problems and solutions are returned"""

    problems, solutions = list(), list()

    max_weight = 100.
    container_shape = MultiPolygon([(((0, 0), (0, 5), (7, 5), (7, 2), (4, 2), (4, 0)),
                                     [((3.2, 2.5), (3.2, 2.7), (3.4, 2.7), (3.4, 2.5)),
                                      ((3.8, 1.8), (3.8, 2.0), (3.99, 2.0), (3.99, 1.8)),
                                      ((3.8, 1.0), (3.8, 1.2), (3.99, 1.2), (3.99, 1.0))])])
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0, 0), (0, 1.5), (1.5, 1.5), (1.5, 0)]), 10., 10.),
             Item(Polygon([(0, 0), (0, 1.5), (1.5, 1.5), (1.5, 0)]), 10., 10.),
             Item(Polygon([(0, 0), (0, 1.5), (1.5, 1.5), (1.5, 0)]), 10., 10.),
             Item(Polygon([(0, 0), (0, 1.), (1., 1.), (1., 0)]), 5., 5.),
             Item(Polygon([(0, 0), (0, 0.4), (1.5, 0.4), (1.5, 0)]), 8., 7.),
             Item(Polygon([(0, 0), (0, 0.4), (1.5, 0.4), (1.5, 0)]), 8., 7.),
             Item(Polygon([(0, 0), (0, 0.4), (1.5, 0.4), (1.5, 0)]), 8., 7.),
             ]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    return problems, [str(i + 1) for i in range(len(problems))], solutions

def create_problems(problem_type, can_print=False):
    return create_knapsack_packing_problems_with_manual_solutions(can_print)


def execute_algorithm_with_params(params):
    # unpack the algorithm and its parameters
    algorithm, algorithm_name, problem, show_solution_plot, solution_plot_save_path, calculate_times, calculate_value_evolution = params

    start_time = time.time()
    value_evolution = None
    times_dict = None
    if calculate_value_evolution:
        if algorithm == evolutionary.solve_problem:
            if calculate_times:
                solution, times_dict, value_evolution = algorithm(problem, calculate_times=calculate_times, return_population_fitness_per_generation=calculate_value_evolution)
            else:
                solution, value_evolution = algorithm(problem, calculate_times=calculate_times, return_population_fitness_per_generation=calculate_value_evolution)
        else:
            if calculate_times:
                solution, times_dict, value_evolution = algorithm(problem, calculate_times=calculate_times, return_value_evolution=calculate_value_evolution)
            else:
                solution, value_evolution = algorithm(problem, calculate_times=calculate_times, return_value_evolution=calculate_value_evolution)
    elif calculate_times:
        solution, times_dict = algorithm(problem, calculate_times=calculate_times)
    else:
        solution = algorithm(problem)
    elapsed_time = get_time_since(start_time)

    if solution and (show_solution_plot or solution_plot_save_path):
        solution.visualize(show_plot=show_solution_plot, save_path=solution_plot_save_path)

    return solution, solution.value, value_evolution, elapsed_time, times_dict


def execute_algorithm(algorithm, algorithm_name, problem, show_solution_plot=False, solution_plot_save_path=None, calculate_times=False, calculate_fitness_stats=False, execution_num=1, process_num=1):
    # encapsulate the algorithm and its parameters in a tuple for each execution (needed for multi-processing)
    param_tuples = [(algorithm, algorithm_name, problem, show_solution_plot, solution_plot_save_path, calculate_times, calculate_fitness_stats) for _ in range(execution_num)]

    solutions, values, value_evolutions, times, time_divisions = list(), list(), list(), list(), list()

    # if possible, perform each execution in a separate CPU process (in parallel)
    if process_num > 1:

        process_pool = Pool(process_num)
        batch_num = ceil(execution_num / process_num)
        for batch in range(batch_num):
            results = process_pool.map(execute_algorithm_with_params, param_tuples[batch * process_num: batch * process_num + process_num])
            batch_solutions, batch_values, batch_value_evolutions, batch_times, batch_time_divisions = [result[0] for result in results], [result[1] for result in results], [result[2] for result in results], [result[3] for result in results], [result[4] for result in results]
            solutions.extend(batch_solutions)
            values.extend(batch_values)
            value_evolutions.extend(batch_value_evolutions)
            times.extend(batch_times)
            time_divisions.extend(batch_time_divisions)
            '''process_pool.terminate()
            process_pool.join()'''

    # perform the calculation sequentially if multi-processing is not allowed
    else:

        for i in range(execution_num):

            solution, value, value_evolution, elapsed_time, time_division = execute_algorithm_with_params(param_tuples[i])
            solutions.append(solution)
            values.append(value)
            value_evolutions.append(value_evolution)
            times.append(elapsed_time)
            time_divisions.append(time_division)

    return solutions, values, value_evolutions, times, time_divisions


def perform_experiments(problem_type, output_dir, load_experiments):
    experiment_file_path = output_dir + "experiments.pickle"

    # data structure where to store all the problems, and the experimental results for each algorithm: solutions, final values and times
    experiment_dict = dict()

    # perform experiments if pre-existing results were not loaded
    if not experiment_dict:

        # given a problem type, create a set of problem instances that are solved (optimally) with manual placements (using actions available for all the algorithms)
        problems, problem_names, manual_solutions = create_problems(problem_type)

        if problems and problem_names and manual_solutions:

            # parameters for the experimentation; note: calculating internal times and value evolution can increase the overall time of algorithms (in a slight, almost neglectible way)
            # execution_num = 10  # 1
            # process_num = 10  # 1
            execution_num = 1
            process_num = 1
            calculate_internal_times = True
            calculate_value_evolution = True

            start_time = time.time()

            # solve each problem with each algorithm
            for i, (problem, problem_name, solution) in enumerate(zip(problems, problem_names, manual_solutions)):

                experiment_dict[problem_name] = {"problem": problem, "manual_solution": solution, "algorithms": dict()}

                # solve the problem with different algorithms, executing each one multiple times to gain statistical significance
                for (algorithm_name, algorithm) in [("Greedy", greedy.solve_problem), ("Reversible", reversible.solve_problem), ("Evolutionary", evolutionary.solve_problem)]:

                    solutions, values, value_evolutions, times, time_divisions = execute_algorithm(algorithm=algorithm, algorithm_name=algorithm_name, problem=problem, execution_num=execution_num, process_num=process_num, calculate_times=calculate_internal_times, calculate_fitness_stats=calculate_value_evolution)
                    experiment_dict[problem_name]["algorithms"][algorithm_name] = {"solutions": solutions, "values": values, "value_evolutions": value_evolutions, "times": times, "time_divisions": time_divisions}

            # show the total time spent doing experiments (note that significant overhead can be introduced beyond calculation time if plots are shown or saved to files; for strict time measurements, plotting should be avoided altogether)
            print("Total experimental calculation time: {} s".format(round(get_time_since(start_time) / 1000.), 3))

    if experiment_dict:

        # experiment-saving parameter
        save_experiments = True

        # if possible, save the experiments to a binary file
        if not load_experiments and save_experiments:
            with open(experiment_file_path, "wb") as file:
                pickle.dump(experiment_dict, file)

        # visualization/saving parameters
        show_problem_stats = False
        save_problem_stats = False  # True
        show_manual_solution_plots = False
        save_manual_solution_plots = False  # True
        show_algorithm_solution_plots = True
        save_algorithm_solution_plots = False  # True
        show_value_evolution_plots = False
        save_value_evolution_plots = False  # True
        show_time_division_plots = False
        save_time_division_plots = False  # True
        show_algorithm_comparison = False
        save_algorithm_comparison = False  # True
        show_aggregated_result_tables = False
        save_aggregated_result_tables = False  # True

        # show/save the results of the experiments
        visualize_and_save_experiments(experiment_dict, output_dir, can_plots_show_value_and_weight=problem_type == KNAPSACK_PACKING_PROBLEM_TYPE, show_problem_stats=show_problem_stats, save_problem_stats=save_problem_stats, show_manual_solution_plots=show_manual_solution_plots, save_manual_solution_plots=save_manual_solution_plots, show_algorithm_solution_plots=show_algorithm_solution_plots, save_algorithm_solution_plots=save_algorithm_solution_plots, show_value_evolution_plots=show_value_evolution_plots, save_value_evolution_plots=save_value_evolution_plots, show_time_division_plots=show_time_division_plots, save_time_division_plots=save_time_division_plots, show_algorithm_comparison=show_algorithm_comparison, save_algorithm_comparison=save_algorithm_comparison, show_aggregated_result_tables=show_aggregated_result_tables, save_aggregated_result_tables=save_aggregated_result_tables)

    else:
        print("The experiments cannot be performed (there are no problems available).")


def visualize_and_save_experiments(experiment_dict, output_dir, can_plots_show_value_and_weight=True, show_problem_stats=False, save_problem_stats=True, show_manual_solution_plots=False, save_manual_solution_plots=True, show_algorithm_solution_plots=False, save_algorithm_solution_plots=True, show_value_evolution_plots=False, save_value_evolution_plots=True, show_time_division_plots=False, save_time_division_plots=True, show_algorithm_comparison=False, save_algorithm_comparison=True, show_aggregated_result_tables=True, save_aggregated_result_tables=True):
    # if needed, save statistics about the problems and their manual solutions
    if show_problem_stats or save_problem_stats:
        problem_names = list(experiment_dict.keys())
        fields = ["Item num.", "Opt. % item num. in cont.", "Opt. % item value in cont.", "Item weight % of max weight", "Item area % of max area", "Cont. weight satur. %", "Cont. area satur. %"]
        data_frame = pd.DataFrame(index=problem_names, columns=fields)
        for problem_name in experiment_dict.keys():
            problem = experiment_dict[problem_name]["problem"]
            solution = experiment_dict[problem_name]["manual_solution"]
            if type(solution) == Solution:
                problem_results = [
                    len(problem.items),
                    round(len(solution.placed_items) / len(problem.items) * 100, 2),
                    round(sum([problem.items[item_index].value for item_index in solution.placed_items.keys()]) / sum([item.value for item in problem.items.values()]) * 100, 2),
                    round(sum([item.weight for item in problem.items.values()]) / problem.container.max_weight * 100, 2),
                    round(sum([item.shape.area for item in problem.items.values()]) / problem.container.shape.area * 100, 2),
                    round(sum([problem.items[item_index].weight for item_index in solution.placed_items.keys()]) / problem.container.max_weight * 100, 2),
                    round(sum([problem.items[item_index].shape.area for item_index in solution.placed_items.keys()]) / problem.container.shape.area * 100, 2),
                ]
                data_frame.loc[problem_name] = problem_results
        if len(data_frame) > 0:
            data_frame.index = [("Problem " + name if len(name) < 5 else name) for name in data_frame.index]
            min_row = data_frame.min()
            max_row = data_frame.max()
            mean_row = data_frame.mean()
            std_row = data_frame.std()
            data_frame.loc["Min"] = round(min_row, 2)
            data_frame.loc["Max"] = round(max_row, 2)
            data_frame.loc["Mean"] = round(mean_row, 2)
            data_frame.loc["Std"] = round(std_row, 2)
            data_frame.loc["Std / (max - min) %"] = round(std_row / (max_row - min_row) * 100, 2)
            if show_problem_stats:
                print(data_frame.to_string())
            if save_problem_stats:
                data_frame.to_excel(output_dir + "problem_stats.xlsx")
                data_frame.to_latex(output_dir + "problem_stats.tex")

    # create the problem results directory (if not done yet)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # for each problem's results, show or save plots if needed
    if show_manual_solution_plots or save_manual_solution_plots or show_algorithm_solution_plots or save_algorithm_solution_plots or show_algorithm_comparison or save_algorithm_comparison:
        for problem_name in experiment_dict.keys():

            problem, manual_solution, algorithm_dict = experiment_dict[problem_name].values()

            # create a subdirectory to store the solutions of the problem (if not done yet)
            problem_dir_path = output_dir + problem_name + "/"
            if not os.path.exists(problem_dir_path):
                os.mkdir(problem_dir_path)

            plotted_problem_name = "Problem " + problem_name if len(problem_name) < 5 else problem_name

            # if needed, show/save a plot of the initial state (empty solution), and the final state of a manual solution
            if show_manual_solution_plots or save_manual_solution_plots:
                empty_solution = Solution(problem)
                empty_solution.visualize(title_override=plotted_problem_name + " - Initial state", show_plot=show_manual_solution_plots, save_path=problem_dir_path + "empty_solution.png" if save_manual_solution_plots else None, show_item_value_and_weight=can_plots_show_value_and_weight, show_value_weight_ratio_bar=can_plots_show_value_and_weight)
                if type(manual_solution) == Solution:
                    manual_solution.visualize(title_override=plotted_problem_name + " - Manual solution", show_plot=show_manual_solution_plots, save_path=problem_dir_path + "manual_solution.png" if save_manual_solution_plots else None, show_item_value_and_weight=can_plots_show_value_and_weight, show_value_weight_ratio_bar=can_plots_show_value_and_weight)

            # if required, show/save plots of the solutions of each algorithm
            if show_algorithm_solution_plots or save_algorithm_solution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    for i, solution in enumerate(subdict["solutions"]):
                        solution.visualize(title_override=plotted_problem_name + " - " + algorithm_name + " solution", show_plot=show_algorithm_solution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_solution.png" if save_algorithm_solution_plots else None, show_item_value_and_weight=can_plots_show_value_and_weight, show_value_weight_ratio_bar=can_plots_show_value_and_weight)

            # if required, show/save plots of the value evolution of each algorithm
            if show_value_evolution_plots or save_value_evolution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    for i, value_evolution in enumerate(subdict["value_evolutions"]):
                        if value_evolution:
                            if type(value_evolution) == list and type(value_evolution[0]) == list:
                                visualize_boxplot_for_data_sequence(data_lists=value_evolution, title=plotted_problem_name + " - Population fitness per generation", show_plot=show_value_evolution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_fitness_evolution.png" if save_value_evolution_plots else None)
                            else:
                                visualize_plot(values=value_evolution, title=plotted_problem_name + " - " + algorithm_name + " solution value per iteration", show_plot=show_value_evolution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_value_evolution.png" if save_value_evolution_plots else None)

            # if required, show/save plots of the time division in tasks of each algorithm
            if show_time_division_plots or save_time_division_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    for i, time_division in enumerate(subdict["time_divisions"]):
                        visualize_bar_plot(values=[value_pair[0] for value_pair in time_division.values()], labels=[add_newlines_by_spaces(label, 7) for label in list(time_division.keys())], title="Problem " + problem_name + " - " + algorithm_name + " time per task (milliseconds)", show_plot=show_algorithm_solution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_time_division.png" if save_algorithm_solution_plots else None)

            # if needed, show/save plots that compare the value and time of each algorithm considering multiple executions
            if show_algorithm_comparison or save_algorithm_comparison:
                visualize_boxplot_for_data_sequence(data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["values"] for algo_name in experiment_dict[problem_name]["algorithms"].keys()], title="Problem " + problem_name + " - Solution value by algorithm", labels=experiment_dict[problem_name]["algorithms"].keys(), show_plot=show_algorithm_comparison, save_path=problem_dir_path + "value_comparison.png" if save_algorithm_comparison else None)
                visualize_boxplot_for_data_sequence(data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["times"] for algo_name in experiment_dict[problem_name]["algorithms"].keys()], title="Problem " + problem_name + " - Computational time (milliseconds) by algorithm", labels=experiment_dict[problem_name]["algorithms"].keys(), y_scale_override="log", show_plot=show_algorithm_comparison, save_path=problem_dir_path + "time_comparison.png" if save_algorithm_comparison else None)

    # if needed, save tables with an aggregation of the value and time results of the executions of each problem (or just show them)
    if show_aggregated_result_tables or save_aggregated_result_tables:
        problem_names = list(experiment_dict.keys())
        algorithm_names = [algo_name for algo_name in experiment_dict[problem_names[0]]["algorithms"].keys()]
        fields = ["mean", "std", "min", "med", "max"]
        for concept in ["value", "time"]:
            algo_field_tuples = [(algo_name, field) for algo_name in algorithm_names for field in fields]
            if concept == "value":
                algo_field_tuples += [("Manual", "optim.")]
            multi_index = pd.MultiIndex.from_tuples(algo_field_tuples, names=["Algorithm", "Statistic"])
            data_frame = pd.DataFrame(index=problem_names, columns=multi_index)
            for problem_name in experiment_dict.keys():
                problem_results = list()
                for algo_name in algorithm_names:
                    mean, std, min_, median, max_ = get_stats(experiment_dict[problem_name]["algorithms"][algo_name][concept + "s"], 2 if concept == "value" else 0)
                    problem_results.extend([mean, std, min_, median, max_])
                if concept == "value":
                    if type(experiment_dict[problem_name]["manual_solution"]) == Solution:
                        problem_results.append(experiment_dict[problem_name]["manual_solution"].value)
                    else:
                        problem_results.append(experiment_dict[problem_name]["manual_solution"])
                data_frame.loc[problem_name] = problem_results
            data_frame.index = [("Problem " + name if len(name) < 5 else name) for name in data_frame.index]
            if show_aggregated_result_tables:
                print("{} results:\n{}\n".format(concept.capitalize(), data_frame.to_string()))
            if save_aggregated_result_tables:
                data_frame.to_excel(output_dir + concept + "_results.xlsx")
                data_frame.to_latex(output_dir + concept + "_results.tex")


def main():

    """Main function"""

    # set the type of problem to solve and the output directory
    problem_type, output_dir = KNAPSACK_PACKING_PROBLEM_TYPE, KNAPSACK_PACKING_PROBLEM_DIR
    # problem_type, output_dir = PACKING_PROBLEM_TYPE, PACKING_PROBLEM_DIR

    # whether it should be attempted to load existing experiments, and avoid running new ones
    load_experiments = False

    # perform (or just load and show) a set of experiments
    perform_experiments(problem_type, output_dir, load_experiments)


if __name__ == "__main__":
    main()
