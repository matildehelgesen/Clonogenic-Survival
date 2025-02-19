import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class SurvivalAnalysis:
    class SurvivalData:
        def __init__(self, raw_data, doses, num_plated):
            """
            Initializes the survival data.

            Parameters:
            - raw_data: List of raw colony counts for each dose.
            - doses: List or array of doses.
            - num_plated: List or array of number of cells plated per dose.
            """
            self.raw_data = raw_data
            self.doses = np.array(doses)
            self.num_plated = np.array(num_plated)
            self.NB = np.array([np.nanmean(data) for data in raw_data])
            self.d_NB = np.array([self.delta_NB(np.nanmean(data), data) for data in raw_data])
            self.PE = self.calculate_PE(np.nanmean(raw_data[0]))
            self.F = self.calculate_F()

        def calculate_PE(self, Nc):
            """
            Calculates the plating efficiency.
            """
            PE = Nc / self.num_plated[0]
            print(PE)
            return PE

        def calculate_F(self):
            """
            Calculates the apparent survival fraction.
            """
            NEB = self.num_plated * self.PE
            return self.NB / NEB

        def delta_NB(self, NB, raw):
            """
            Calculates the uncertainty in NB.
            """
            n = len(raw)
            factor = 1 / (n * (n - 1))
            return np.sqrt(factor * np.sum((raw - NB) ** 2))

    class SurvivalModel:
        def __init__(self, survival_data, xiN):
            """
            Initializes the survival model.
            Parameters:
            - survival_data: An instance of SurvivalData.
            - xiN: Array representing the fraction of CFU per type.
            """
            self.survival_data = survival_data
            self.xiN = xiN / np.sum(xiN)  # Fraction of CFU per type
            self.i = np.array([1, 2, 3])  # Singlet, doublet, triplet
            self.M = self.calculate_M()
            self.S_approx = self.calculate_S_approximation()
            self.S_solution = self.solve_S_numerical()

        def calculate_M(self):
            """
            Calculates the mean multiplicity.
            """
            return np.sum(self.xiN * (self.i)) / np.sum(self.xiN)

        def calculate_S_approximation(self):
            """
            Calculates the approximate survival using the multiplicity.
            """
            F = self.survival_data.F
            M = self.M
            return (M - np.sqrt((M ** 2) - 4 * (M - 1) * F)) / (2 * (M - 1))

        def S_equation(self, S, F):
            """
            Define the equation F = sum(xi * (1 - (1 - S)^i)) for a given S.
            """
            return F - np.sum(self.xiN * (1 - (1 - S) ** self.i))

        def solve_S_numerical(self):
            """
            Solves the survival fraction numerically.
            """
            S_solution = np.zeros(len(self.survival_data.doses))
            for idx, F in enumerate(self.survival_data.F):
                S_guess = 1.0
                S_solution[idx], _, _, _ = fsolve(self.S_equation, S_guess, args=(F), full_output=True)
            return S_solution

    class SurvivalCurvePlotter:
        def __init__(self, survival_data, survival_model, title):
            """
            Initializes the plotter.

            Parameters:
            - survival_data: An instance of SurvivalData.
            - survival_model: An instance of SurvivalModel.
            - title: Title for the plot.
            """
            self.survival_data = survival_data
            self.survival_model = survival_model
            self.title = title

        def plot(self):
            """
            Plots the survival curves using apparent, approximate, and numerical survival.
            """
            doses = self.survival_data.doses
            F = self.survival_data.F
            S_approx = self.survival_model.S_approx
            S_solution = self.survival_model.S_solution

            plt.plot(doses, F, label='Apparent Survival', marker='o')
            plt.plot(doses, S_approx, label='Approximate Survival', linestyle='--')
            plt.plot(doses, S_solution, label='Numerical Survival', linestyle='-.')
            plt.yscale('log')
            plt.xlabel('Dose (Gy)')
            plt.ylabel('Survival Fraction')
            plt.title(self.title)
            plt.grid(True)
            plt.legend()
            plt.show()

    class SurvivalDataWriter:
        def __init__(self, survival_data, survival_model, filename):
            """
            Initializes the data writer.

            Parameters:
            - survival_data: An instance of SurvivalData.
            - survival_model: An instance of SurvivalModel.
            - filename: The filename to write the data to.
            """
            self.survival_data = survival_data
            self.survival_model = survival_model
            self.filename = filename

        def write_data(self):
            """
            Writes the survival data and model results to a file.
            """
            with open(self.filename, "w") as file:
                #self.print_array(self.survival_data.F, "Apparent Survival", file)
                self.print_array(self.survival_model.S_approx, "Approximate Survival", file)
                self.print_array(self.survival_model.S_solution, "Numerical Survival", file)
                #self.print_array(self.survival_data.d_NB, "DeltaNB", file)
                #self.print_array(self.survival_data.d_F, "DeltaF", file)
                #self.print_array(self.survival_data.d_S_approx, "DeltaNB", file)

        def print_array(self, array, name, file):
            """
            Prints the array to both the console and a file.
            """
            print(f"---------- {name} ----------")
            file.write(f"{name}\n")
            for idx in range(len(array)):
                file.write(f"{array[idx]} ")
                print(array[idx])
            file.write("\n")
            print()
