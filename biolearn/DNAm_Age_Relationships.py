import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class DNAMethylationAnalyzer:
    def __init__(self, metadata: pd.DataFrame, methylation_data: pd.DataFrame):
        self.metadata = metadata
        self.methylation_data = methylation_data

    def preprocess_data(self):
        """
        Preprocesses the methylation data by handling missing values and ensuring data consistency.
        """
        # Drop rows with all NaN values (CpG sites with no data)
        self.methylation_data.dropna(how='all', inplace=True)
        # Fill NaN values with the mean value for each CpG site
        self.methylation_data.fillna(self.methylation_data.mean(), inplace=True)

    def identify_stable_cpg_sites(self, threshold=0.01):
        """
        Identifies stable CpG sites based on variance.
        A CpG site is considered stable if its variance is below the given threshold.
        
        :param threshold: Variance threshold to determine stability (default: 0.01)
        :return: List of stable CpG site IDs
        """
        # Calculate variance for each CpG site (each row)
        variances = self.methylation_data.var(axis=1)
        # Identify CpG sites with variance below the threshold
        stable_cpg_sites = variances[variances < threshold].index.tolist()        
        return stable_cpg_sites

    def identify_significant_cpg_sites(self, group_field: str = None, group1: str = None, group2: str = None, p_value_threshold=0.05):
        """
        Identifies significant CpG sites between two groups using a t-test.
        
        :param group_field: The metadata field used to define groups (e.g., 'Disease State')
        :param group1: The value of the group_field for group 1 (e.g., 'None')
        :param group2: The value of the group_field for group 2 (e.g., 'COVID')
        :param p_value_threshold: The p-value threshold for significance (default: 0.05)
        :return: List of significant CpG site IDs
        """
        # Get sample IDs for each group
        group1_samples = self.metadata[self.metadata[group_field] == group1].index.tolist()
        group2_samples = self.metadata[self.metadata[group_field] == group2].index.tolist()

        significant_cpg_sites = []

        # Iterate over each CpG site and perform a t-test
        for cpg_site, row in self.methylation_data.iterrows():
            # Extract methylation values for the CpG site across group1 and group2 samples
            group1_values = row[group1_samples].dropna()
            group2_values = row[group2_samples].dropna()

            # Perform t-test
            if len(group1_values) > 1 and len(group2_values) > 1:  # Ensure enough data points
                t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
                
                # Check if p-value is below the threshold
                if p_value < p_value_threshold:
                    significant_cpg_sites.append(cpg_site)

        return significant_cpg_sites

    def plot_cpg_against_age(self, cpg_site: str):
        """
        Plots the methylation level of a specific CpG site against age and explores linear and non-linear relationships.
        
        :param cpg_site: The CpG site ID to plot
        """
        # Extract the methylation values for the specified CpG site
        cpg_data = self.methylation_data.set_index('cpgSite').loc[cpg_site]
        # Merge with metadata to get age information
        merged_data = pd.merge(cpg_data.reset_index(), self.metadata, left_on='index', right_on='SampleID')

        # Extract age and methylation values
        ages = merged_data['Age'].values.reshape(-1, 1)
        methylation_levels = merged_data[cpg_site].values

        # Linear Relationship
        linear_model = LinearRegression()
        linear_model.fit(ages, methylation_levels)
        linear_pred = linear_model.predict(ages)

        plt.figure(figsize=(12, 6))
        plt.scatter(ages, methylation_levels, color='blue', label='Data')
        plt.plot(ages, linear_pred, color='red', label='Linear Fit')
        plt.xlabel('Age')
        plt.ylabel('Methylation Level')
        plt.title(f'CpG Site {cpg_site} Methylation vs Age (Linear Relationship)')
        plt.legend()
        plt.show()

        # Non-Linear Relationship (Polynomial Fit)
        poly = PolynomialFeatures(degree=2)
        ages_poly = poly.fit_transform(ages)
        poly_model = LinearRegression()
        poly_model.fit(ages_poly, methylation_levels)
        poly_pred = poly_model.predict(ages_poly)

        plt.figure(figsize=(12, 6))
        plt.scatter(ages, methylation_levels, color='blue', label='Data')
        plt.plot(ages, poly_pred, color='green', label='Polynomial Fit (Degree 2)')
        plt.xlabel('Age')
        plt.ylabel('Methylation Level')
        plt.title(f'CpG Site {cpg_site} Methylation vs Age (Non-Linear Relationship)')
        plt.legend()
        plt.show()

    def visualize_correlation_with_age(self, cpg_sites: list):
        """
        Visualizes the correlation between methylation levels of specific CpG sites and age.
        
        :param cpg_sites: List of CpG site IDs to visualize
        """
        correlations = {}
        for cpg_site in cpg_sites:
            # Extract the methylation values for the specified CpG site
            cpg_data = self.methylation_data.set_index('cpgSite').loc[cpg_site]
            # Merge with metadata to get age information
            merged_data = pd.merge(cpg_data.reset_index(), self.metadata, left_on='index', right_on='SampleID')

            # Extract age and methylation values
            ages = merged_data['Age'].values
            methylation_levels = merged_data[cpg_site].values

            # Calculate Pearson correlation coefficient
            corr, _ = pearsonr(ages, methylation_levels)
            correlations[cpg_site] = corr

            # Plotting the relationship
            plt.figure(figsize=(12, 6))
            plt.scatter(ages, methylation_levels, color='blue', label=f'Data (Correlation: {corr:.2f})')
            plt.xlabel('Age')
            plt.ylabel('Methylation Level')
            plt.title(f'CpG Site {cpg_site} Methylation vs Age (Correlation: {corr:.2f})')
            plt.legend()
            plt.show()

        return correlations