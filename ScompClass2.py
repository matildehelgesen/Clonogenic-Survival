import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import statsmodels.api as sm

class SurvivalComparison:
    def __init__(self, doses):
        """
        Initializes the SurvivalComparison class for plotting and fitting data from multiple experiments.

        Parameters:
        - doses: Array of doses.
        """
        self.doses = doses
        self.experiments = []
        self.S_num_all = [[] for _ in range(len(doses))]  # A list for each dose to store S_num values

    def extract_data(self, filename):
        """
        Extracts data from a file.

        Parameters:
        - filename: The filename to extract data from.
        """
        def convert_to_float(val):
            try:
                return float(val)
            except ValueError:
                return np.nan

        with open(filename, "r") as file:
            file.readline()
            S_approx = np.array([convert_to_float(x) for x in file.readline().split()])
            file.readline()
            S_num = np.array([convert_to_float(x) for x in file.readline().split()])
            file.readline()
            d_NB = np.array([convert_to_float(x) for x in file.readline().split()])

        return S_approx, S_num, d_NB

    def add_experiment(self, filename, label):
        """
        Adds an experiment to the comparison.

        Parameters:
        - filename: The filename of the experiment data.
        - label: Label for the experiment.
        """
        S_approx, S_num, d_NB = self.extract_data(filename)
        self.experiments.append({
            "filename": filename,
            "label": label,
            "S_approx": S_approx,
            "S_num": S_num,
            "d_NB": d_NB
        })
        for i, s_value in enumerate(S_num):
            self.S_num_all[i].append(s_value)

    def calculate_mean_and_sem(self):
        """
        Calculates the mean and standard error of the mean (SEM) for the S_num values at each dose.
        """
        mean_data = np.array([np.nanmean(dose_values) for dose_values in self.S_num_all])
        sem_data = np.array([np.nanstd(dose_values) / np.sqrt(len(dose_values)) for dose_values in self.S_num_all])
        return mean_data, sem_data

    def plot_experiments(self, title):
        """
        Plots all added experiments with error bars.

        Parameters:
        - title: Title for the plot.
        """
        plt.figure()
        for experiment in self.experiments:
            plt.errorbar(self.doses, experiment["S_num"], fmt='.-', label=experiment["label"], capsize=5)
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_combined_mean(self, title):
        """
        Plots the combined mean of numerical survival data for all experiments with error bars.
        Parameters:
        - title: Title for the plot.
        """
        mean, sem = self.calculate_mean_and_sem()
        plt.figure()
        plt.errorbar(self.doses, mean, yerr=sem, fmt='.-', label='Combined Mean (Numerical)', capsize=5)
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_combined_lq_fit_with_confidence2(self, title):
        """
        Fits and plots the linear-quadratic (LQ) model using statsmodels OLS for the combined mean of numerical survival data.
        Parameters:
        - title: Title for the plot.
        diffrence: includes the 0 gy values in the optimization.
        """
        # Calculate mean and SEM
        mean, sem = self.calculate_mean_and_sem()

        # Transform survival data: ln(Survival Fraction)
        ln_survival = np.log(mean)

        # Define the design matrix for the linear model: [Dose, Dose^2]
        D = np.array(self.doses)
        X = np.column_stack((-D, -D**2))  # Linear terms corresponding to -alpha*D and -beta*D^2
        X = sm.add_constant(X)  # Add intercept term if needed

        # Fit the model using OLS
        model = sm.OLS(ln_survival, X)
        results = model.fit()

        # Extract fitted parameters and their uncertainties
        alpha, beta = results.params[1], results.params[2]
        alpha_se, beta_se = results.bse[1], results.bse[2]

        # Print alpha and beta with uncertainties
        print(f"Alpha: {alpha:.4f} ± {alpha_se:.4f}")
        print(f"Beta: {beta:.4f} ± {beta_se:.4f}")
        print(results.summary())

        # Generate LQ model predictions for plotting
        x_fit = np.linspace(min(self.doses), max(self.doses), 1000)
        X_fit = sm.add_constant(np.column_stack((-x_fit, -x_fit**2)))
        pred = results.get_prediction(X_fit)
        pred_summary = pred.summary_frame(alpha=0.01)  # 99% confidence level

        y_fit = np.exp(pred_summary['mean'])  # Predicted mean
        y_fit_lower = np.exp(pred_summary['mean_ci_lower'])  # Lower bound of 99% CI
        y_fit_upper = np.exp(pred_summary['mean_ci_upper'])  # Upper bound of 99% CI

        # Plot the data and the fit
        plt.figure()
        plt.errorbar(self.doses, mean, yerr=sem, fmt='.', label='Combined Mean (Numerical Data)', capsize=5)
        plt.plot(x_fit, y_fit, linestyle='--', label='LQ Model Fit (OLS)')
        #plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.3, label='99% Prediction Interval')
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_combined_lq_fit_with_confidence(self, title, Type):
        """
        Fits and plots the linear-quadratic (LQ) model using statsmodels OLS for all experiments in the instance.

        Parameters:
        - title: Title for the plot.
        """
        # Collect all S_approx data from all experiments into a single array
        all_survival_data = []
        all_doses = []
        for experiment in self.experiments:
            all_survival_data.extend(experiment["S_approx"])
            all_doses.extend(self.doses)

        all_survival_data = np.array(all_survival_data)
        all_doses = np.array(all_doses)

        # Transform survival data: ln(Survival Fraction)
        ln_survival = np.log(all_survival_data)

         # Define the design matrix for the linear model: [Dose, Dose^2]
        X = np.column_stack((-all_doses, -all_doses**2))  # Linear terms corresponding to -alpha*D and -beta*D^2
        X = sm.add_constant(X)  # Add intercept term if needed

        # Fit the model using OLS
        model = sm.OLS(ln_survival, X)
        results = model.fit()

        # Extract fitted parameters and their uncertainties
        alpha, beta = results.params[1], results.params[2]
        alpha_se, beta_se = results.bse[1], results.bse[2]

        # Get 99% confidence intervals for alpha and beta
        conf_intervals = results.conf_int(alpha=0.01)  # alpha=0.01 for 99% confidence level
        alpha_ci = conf_intervals[1]  # Confidence interval for alpha
        beta_ci = conf_intervals[2]  # Confidence interval for beta

        # Print alpha and beta with 99% confidence intervals
        print(f"Alpha: {alpha:.4f} (99% CI: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}])")
        print(f"Beta: {beta:.4f} (99% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}])")
        print(results.summary())


        # Generate LQ model predictions for plotting
        x_fit = np.linspace(min(self.doses), max(self.doses), 1000)
        X_fit = sm.add_constant(np.column_stack((-x_fit, -x_fit**2)))
        pred = results.get_prediction(X_fit)
        pred_summary = pred.summary_frame(alpha=0.01)  # 99% confidence level

        y_fit = np.exp(-(alpha * x_fit + beta * x_fit**2))
        #y_fit = np.exp(pred_summary['mean'])  # Predicted mean
        y_fit_lower = np.exp(pred_summary['mean_ci_lower'])  # Lower bound of 99% CI
        y_fit_upper = np.exp(pred_summary['mean_ci_upper'])  # Upper bound of 99% CI

        # Plot the data and the fit
        plt.figure()
        plt.errorbar(self.doses, np.nanmean(self.S_num_all, axis=1), yerr=np.nanstd(self.S_num_all, axis=1),
                     fmt='.', label=f'Mean mesurement and SEM, {Type}', capsize=5)
        plt.plot(x_fit, y_fit, linestyle='--', label=f'LQ Model Fit, {Type}')
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.3, label=f'99% PI, {Type}')
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_combined_lq_fit_with_confidence2(self, title):
        """
        Fits and plots the linear-quadratic (LQ) model using statsmodels OLS for all experiments in the instance.

        Parameters:
        - title: Title for the plot.
        """
        # Collect all S_approx data from all experiments into a single array
        all_survival_data = []
        all_doses = []
        for experiment in self.experiments:
            for dose, survival in zip(self.doses, experiment["S_approx"]):
                if not np.isnan(survival):
                    all_survival_data.append(survival)
                    all_doses.append(dose)

        all_survival_data = np.array(all_survival_data)
        all_doses = np.array(all_doses)

        # Remove the 0 Gy points from the data
        non_zero_indices = all_doses > 0
        all_survival_data = all_survival_data[non_zero_indices]
        all_doses = all_doses[non_zero_indices]

        # Transform survival data: ln(Survival Fraction)
        ln_survival = np.log(all_survival_data)

        # Define the design matrix for the linear model: [Dose, Dose^2]
        X = np.column_stack((-all_doses, -all_doses**2))  # Linear terms corresponding to -alpha*D and -beta*D^2
        X = sm.add_constant(X)  # Add intercept term if needed

        # Fit the model using OLS
        model = sm.OLS(ln_survival, X)
        results = model.fit()

        # Extract fitted parameters and their uncertainties
        alpha, beta = results.params[1], results.params[2]
        alpha_se, beta_se = results.bse[1], results.bse[2]

        # Get 99% confidence intervals for alpha and beta
        conf_intervals = results.conf_int(alpha=0.01)  # alpha=0.01 for 99% confidence level
        alpha_ci = conf_intervals[1]  # Confidence interval for alpha
        beta_ci = conf_intervals[2]  # Confidence interval for beta

        # Print alpha and beta with 99% confidence intervals
        print(f"Alpha: {alpha:.4f} (99% CI: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}])")
        print(f"Beta: {beta:.4f} (99% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}])")
        print(results.summary())

        # Generate LQ model predictions for plotting
        x_fit = np.linspace(min(self.doses), max(self.doses), 1000)
        X_fit = sm.add_constant(np.column_stack((-x_fit, -x_fit**2)))
        pred = results.get_prediction(X_fit)
        pred_summary = pred.summary_frame(alpha=0.01)  # 99% confidence level


        y_fit = np.exp(-(alpha * x_fit + beta * x_fit**2))
        #y_fit = np.exp(pred_summary['mean'])  # Predicted mean
        y_fit[0] = 1
        y_fit_lower = np.exp(pred_summary['mean_ci_lower'])  # Lower bound of 99% CI
        y_fit_upper = np.exp(pred_summary['mean_ci_upper'])  # Upper bound of 99% CI

        # Plot the data and the fit
        plt.figure()
        plt.errorbar(self.doses, np.nanmean(self.S_num_all, axis=1), yerr=np.nanstd(self.S_num_all, axis=1),
                     fmt='.', label='Combined Mean (Numerical Data)', capsize=5)
        plt.plot(x_fit, y_fit, linestyle='--', label='LQ Model Fit (OLS)')
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.3, label='99% Prediction Interval')
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_multiple_instances(instances, title, instance_labels=None):
        """
        Plots experiments from multiple SurvivalComparison instances in the same plot.

        Parameters:
        - instances: List of SurvivalComparison instances.
        - title: Title for the plot.
        - instance_labels: List of labels for each instance.
        """
        plt.figure()
        for idx, instance in enumerate(instances):
            label_prefix = instance_labels[idx] if instance_labels else f"Instance {idx+1}"
            for experiment in instance.experiments:
                plt.errorbar(instance.doses, experiment["S_approx"], fmt='.-', label=f"{label_prefix}: {experiment['label']}", capsize=5)
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title, loc='center', wrap=True)
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_combined_mean_multiple_instances(instances, title, instance_labels=None, additional_experiment_instance=None):
        """
        Plots the combined mean of numerical survival data from multiple instances in the same plot, with an option to add single experiments from an additional instance.

        Parameters:
        - instances: List of SurvivalComparison instances.
        - title: Title for the plot.
        - instance_labels: List of labels for each instance.
        - additional_experiment_instance: An instance of SurvivalComparison representing additional single experiments to add to the plot.
        """
        plt.figure()
        for idx, instance in enumerate(instances):
            mean, sem = instance.calculate_mean_and_sem()
            label = instance_labels[idx] if instance_labels else f'Mean mesurement and SEM,(Instance {idx+1})'
            plt.errorbar(instance.doses, mean, yerr=sem, fmt='.-', label=label, capsize=5)

        if additional_experiment_instance:
            for experiment in additional_experiment_instance.experiments:
                plt.errorbar(additional_experiment_instance.doses, experiment["S_approx"], fmt='.-', label=experiment["label"], capsize=5)

        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title, loc='center', wrap=True)
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_combined_lq_fit_multiple_instances(instances, title, colors, instance_labels=None, additional_experiment_instance=None,):
        """
        Fits and plots the linear-quadratic (LQ) model for the combined mean of numerical survival data from multiple instances, with an option to add single experiments from an additional instance.

        Parameters:
        - instances: List of SurvivalComparison instances.
        - title: Title for the plot.
        - instance_labels: List of labels for each instance.
        - additional_experiment_instance: An instance of SurvivalComparison representing additional single experiments to add to the plot.
        """

        plt.figure()
        for idx, instance in enumerate(instances):
            all_survival_data = []
            all_doses = []
            for experiment in instance.experiments:
                all_survival_data.extend(experiment["S_approx"])
                all_doses.extend(instance.doses)

            all_survival_data = np.array(all_survival_data)
            all_doses = np.array(all_doses)

            # Transform survival data: ln(Survival Fraction)
            ln_survival = np.log(all_survival_data)
            # Define the design matrix for the linear model: [Dose, Dose^2]
            X = np.column_stack((-all_doses, -all_doses**2))  # Linear terms corresponding to -alpha*D and -beta*D^2
            X = sm.add_constant(X)  # Add intercept term if needed

            # Fit the model using OLS
            model = sm.OLS(ln_survival, X)
            results = model.fit()

            # Extract fitted parameters and their uncertainties
            alpha, beta = results.params[1], results.params[2]
            alpha_se, beta_se = results.bse[1], results.bse[2]

            # Generate LQ model predictions for plotting
            x_fit = np.linspace(min(instance.doses), max(instance.doses), 1000)
            X_fit = sm.add_constant(np.column_stack((-x_fit, -x_fit**2)))
            pred = results.get_prediction(X_fit)
            pred_summary = pred.summary_frame(alpha=0.01)  # 99% confidence level

            y_fit = np.exp(-(alpha * x_fit + beta * x_fit**2))
            #y_fit = np.exp(pred_summary['mean'])  # Predicted mean
            y_fit_lower = np.exp(pred_summary['mean_ci_lower'])  # Lower bound of 99% CI
            y_fit_upper = np.exp(pred_summary['mean_ci_upper'])  # Upper bound of 99% CI

            plt.errorbar(instance.doses, np.nanmean(instance.S_num_all, axis=1), yerr=np.nanstd(instance.S_num_all, axis=1),
                         fmt='.', linewidth= 1.75,  label=f'Mean mesurement and SEM, {instance_labels[idx]}', color = colors[idx], capsize=5)
            plt.plot(x_fit, y_fit, linestyle='--', linewidth= 1.75, label=f"LQ Model Fit {instance_labels[idx]}( Alpha: {alpha:.4f} ± {alpha_se:.4f}, Beta: {beta:.4f} ± {beta_se:.4f}) ", color = colors[idx])
            #plt.axvline(alpha/beta, label = f'alpha/beta, {instance_labels[idx]}', color = colors[idx])
            plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color = colors[idx], alpha=0.2, label=f'99% CI, {instance_labels[idx]}')
        # Plot additional experiments if provided

        if additional_experiment_instance:
            for experiment in additional_experiment_instance.experiments:
                plt.errorbar(
                    additional_experiment_instance.doses,
                    experiment["S_approx"],
                    fmt='.-',
                    label=experiment["label"],
                    capsize=5
                )

        # Finalize plot
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title, loc='center', wrap=True)
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_combined_lq_fit_multiple_instances2(instances, title, instance_labels=None, additional_experiment_instance=None):
        """
        Fits and plots the linear-quadratic (LQ) model for the combined mean of numerical survival data
        from multiple instances on the same plot, with the 0 Gy point excluded from fitting
        and enforced to Survival = 1.

        Parameters:
        - instances: List of SurvivalComparison instances.
        - title: Title for the plot.
        - instance_labels: List of labels for each instance.
        - additional_experiment_instance: An instance of SurvivalComparison representing additional single experiments to add to the plot.
        """
        plt.figure()

        # Iterate through the instances
        for idx, instance in enumerate(instances):
            mean, sem = instance.calculate_mean_and_sem()
            ln_survival = np.log(mean)

            # Define the design matrix for the linear model (no intercept)
            D = np.array(instance.doses)
            X = np.column_stack((-D, -D**2))  # Linear terms corresponding to -alpha*D and -beta*D^2

            # Exclude the 0 Gy point from the fit
            non_zero_indices = D > 0
            D_fit = D[non_zero_indices]
            ln_survival_fit = ln_survival[non_zero_indices]
            X_fit = X[non_zero_indices]

            # Fit the model using OLS
            model = sm.OLS(ln_survival_fit, X_fit)
            results = model.fit()

            # Extract fitted parameters
            alpha, beta = results.params[0], results.params[1]
            alpha_se, beta_se = results.bse[0], results.bse[1]

            # Print alpha and beta with uncertainties
            label = instance_labels[idx] if instance_labels else f'Instance {idx+1}'
            print(f"{label}: Alpha = {alpha:.4f} ± {alpha_se:.4f}, Beta = {beta:.4f} ± {beta_se:.4f}")
            print(results.summary())

            # Generate LQ model predictions
            x_fit = np.linspace(min(instance.doses), max(instance.doses), 1000)
            y_fit = np.exp(-(alpha * x_fit + beta * x_fit**2))  # LQ model
            y_fit[0] = 1  # Explicitly set the point at 0 Gy to Survival = 1

            # Plot data and fits
            plt.errorbar(instance.doses, mean, yerr=sem, fmt='.', label=f'{label}', capsize=5)
            plt.plot(x_fit, y_fit, linestyle='--', label=f'LQ Model Fit ({label})')
            plt.fill_between(
                x_fit,
                np.exp(-(alpha * x_fit + beta * x_fit**2 - alpha_se)),
                np.exp(-(alpha * x_fit + beta * x_fit**2 + alpha_se)),
                color='gray',
                alpha=0.2
            )

        # Plot additional experiments if provided
        if additional_experiment_instance:
            for experiment in additional_experiment_instance.experiments:
                plt.errorbar(
                    additional_experiment_instance.doses,
                    experiment["S_approx"],
                    fmt='.-',
                    label=experiment["label"],
                    capsize=5
                )

        # Finalize plot
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title, loc='center', wrap=True)
        plt.grid(True)
        plt.legend()
        plt.show()

        """
        Fits and plots the linear-quadratic (LQ) model for the combined mean of numerical survival data from multiple instances, with an option to add single experiments from an additional instance.

        Parameters:
        - instances: List of SurvivalComparison instances.
        - title: Title for the plot.
        - instance_labels: List of labels for each instance.
        - additional_experiment_instance: An instance of SurvivalComparison representing additional single experiments to add to the plot.
        """

        plt.figure()
        for idx, instance in enumerate(instances):
            all_survival_data = []
            all_doses = []
            for experiment in instance.experiments:
                all_survival_data.extend(experiment["S_approx"])
                all_doses.extend(instance.doses)

            all_survival_data = np.array(all_survival_data)
            D = np.array(all_doses)

            # Transform survival data: ln(Survival Fraction)
            ln_survival = np.log(all_survival_data)
            # Define the design matrix for the linear model: [Dose, Dose^2]
            X = np.column_stack((-D, -D**2))  # Linear terms corresponding to -alpha*D and -beta*D^2
            X = sm.add_constant(X)  # Add intercept term if needed
            #removing S(0) values from fit.

            non_zero_indices = D > 0
            D_fit = D[non_zero_indices]
            ln_survival_fit = ln_survival[non_zero_indices]
            X_fit = X[non_zero_indices]

            # Fit the model using OLS
            model = sm.OLS(ln_survival_fit, X)
            results = model.fit()

            # Extract fitted parameters and their uncertainties
            alpha, beta = results.params[0], results.params[1]
            alpha_se, beta_se = results.bse[0], results.bse[1]

            # Generate LQ model predictions for plotting
            x_fit = np.linspace(min(instance.doses), max(instance.doses), 1000)
            X_fit = sm.add_constant(np.column_stack((-x_fit, -x_fit**2)))
            pred = results.get_prediction(X_fit)
            pred_summary = pred.summary_frame(alpha=0.01)  # 99% confidence level

            y_fit = np.exp(pred_summary['mean'])  # Predicted mean
            y_fit_lower = np.exp(pred_summary['mean_ci_lower'])  # Lower bound of 99% CI
            y_fit_upper = np.exp(pred_summary['mean_ci_upper'])  # Upper bound of 99% CI

            plt.errorbar(instance.doses, np.nanmean(instance.S_num_all, axis=1), yerr=np.nanstd(instance.S_num_all, axis=1),
                         fmt='.', label='Combined Mean (Numerical Data)', capsize=5)
            plt.plot(x_fit, y_fit, linestyle='--', label=f"LQ Model Fit (OLS)( Alpha: {alpha:.4f} ± {alpha_se:.4f}, Beta: {beta:.4f} ± {beta_se:.4f}) ")
            plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.3, label='99% Prediction Interval')
        # Plot additional experiments if provided
        if additional_experiment_instance:
            for experiment in additional_experiment_instance.experiments:
                plt.errorbar(
                    additional_experiment_instance.doses,
                    experiment["S_approx"],
                    fmt='.-',
                    label=experiment["label"],
                    capsize=5
                )

        # Finalize plot
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title, loc='center', wrap=True)
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_combined_lq_fit_and_mean(instance, title, colors, lq_instance_idx=0, mean_instance_idx=1, instance_labels=None):
        """
        Plots the linear-quadratic (LQ) model for one instance and the mean with SEM for another instance on the same plot.
        """

        lq_instance = instance[lq_instance_idx]
        mean_instance = instance[mean_instance_idx]

        plt.figure()
        all_survival_data = []
        all_doses = []
        for experiment in lq_instance.experiments:
            all_survival_data.extend(experiment["S_approx"])
            all_doses.extend(lq_instance.doses)

        all_survival_data = np.array(all_survival_data)
        all_doses = np.array(all_doses)

        # Transform survival data: ln(Survival Fraction)
        ln_survival = np.log(all_survival_data)
        # Define the design matrix for the linear model: [Dose, Dose^2]
        X = np.column_stack((-all_doses, -all_doses**2))  # Linear terms corresponding to -alpha*D and -beta*D^2
        X = sm.add_constant(X)  # Add intercept term if needed

        # Fit the model using OLS
        model = sm.OLS(ln_survival, X)
        results = model.fit()

        # Extract fitted parameters and their uncertainties
        alpha, beta = results.params[1], results.params[2]
        alpha_se, beta_se = results.bse[1], results.bse[2]

        # Generate LQ model predictions for plotting
        x_fit = np.linspace(min(lq_instance.doses), max(lq_instance.doses), 1000)
        X_fit = sm.add_constant(np.column_stack((-x_fit, -x_fit**2)))
        pred = results.get_prediction(X_fit)
        pred_summary = pred.summary_frame(alpha=0.01)  # 99% confidence level

        y_fit = np.exp(-(alpha * x_fit + beta * x_fit**2))
        #y_fit = np.exp(pred_summary['mean'])  # Predicted mean
        y_fit_lower = np.exp(pred_summary['mean_ci_lower'])  # Lower bound of 99% CI
        y_fit_upper = np.exp(pred_summary['mean_ci_upper'])  # Upper bound of 99% CI

        plt.errorbar(lq_instance.doses, np.nanmean(lq_instance.S_num_all, axis=1), yerr=np.nanstd(lq_instance.S_num_all, axis=1),
                     fmt='.', label= f'Mean mesurement and SEM: {instance_labels[0]}', color = colors[0], capsize=5)
        plt.plot(x_fit, y_fit, linestyle='--',color = colors[0], label=f"LQ Model Fit: {instance_labels[0]} ")
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper,  color = colors[0], alpha=0.3, label=f'99% PI: {instance_labels[0]}')

        # Handle Mean Instance
        mean, sem = mean_instance.calculate_mean_and_sem()
        mean_label = instance_labels[mean_instance_idx] if instance_labels else f'Instance {mean_instance_idx + 1}'

        # Plot only the mean with SEM for the other instance
        plt.errorbar(mean_instance.doses, mean, yerr=sem, fmt='.-', color = colors[1], label=f'Mean mesurement and SEM: {instance_labels[1]}', capsize=5)

        # Finalize plot
        plt.yscale('log')
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Survival Fraction')
        plt.title(title, loc='center', wrap=True)
        plt.grid(True)
        plt.legend()
        plt.show()
