import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

class DNAm_age_relationships:
    def __init__(self, datasets):
        """
        Initializes the MethylationAnalysisTool with multiple datasets.

        Parameters:
        - datasets (list of tuples): Each tuple contains two DataFrames, (data_dnam, data_meta).
            * data_dnam: DNAm data with CpG sites as rows and samples as columns.
            * data_meta: Metadata with sample IDs as index, containing 'age' and 'sex' columns.
        """
        self.datasets = datasets

    def identify_stable_cpg_sites(self, threshold=0.01):
        """
        Identifies stable CpG sites in each dataset based on variance.

        Parameters:
        - threshold: Variance threshold to determine stability (default: 0.01)
        """
        stable_sites_summary = []
        for i, (data_dnam, _) in enumerate(self.datasets):
            variances = data_dnam.var(axis=1)
            stable_sites = variances[variances < threshold].index.tolist()
            stable_sites_summary.append({'Dataset': f'Dataset {i+1}', 'Stable CpG Sites': stable_sites})
        return stable_sites_summary

    def identify_significant_cpg_sites(self, group_field, group1, group2, p_value_threshold=0.05):
        """
        Identifies significant CpG sites between two groups in each dataset using a t-test.

        Parameters:
        - group_field: The metadata field to define groups (e.g., 'Disease State')
        - group1: Group 1 value
        - group2: Group 2 value
        - p_value_threshold: Threshold for significance (default: 0.05)
        """
        significant_sites_summary = []
        for i, (data_dnam, data_meta) in enumerate(self.datasets):
            group1_samples = data_meta[data_meta[group_field] == group1].index
            group2_samples = data_meta[data_meta[group_field] == group2].index
            significant_sites = []
            for cpg_site in data_dnam.index:
                group1_values = data_dnam.loc[cpg_site, group1_samples].dropna()
                group2_values = data_dnam.loc[cpg_site, group2_samples].dropna()
                if len(group1_values) > 1 and len(group2_values) > 1:
                    t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
                    if p_value < p_value_threshold:
                        significant_sites.append(cpg_site)
            significant_sites_summary.append({'Dataset': f'Dataset {i+1}', 'Significant CpG Sites': significant_sites})
        return significant_sites_summary

    def compute_statistics_by_age_group(self, cpg_site, age_bins=5):
        """
        Computes statistics (mean, median, std, etc.) for methylation levels by age group and sex.

        Parameters:
        - cpg_site: CpG site ID to compute statistics for.
        - age_bins: Number of age bins for grouping.

        Returns:
        - stats_df: DataFrame containing computed statistics.
        - combined_data: DataFrame with added 'age_group' column.
        """
        combined_data = []
        sex_mapping = {1: 'M', 2: 'F', '1': 'M', '2': 'F', 'M': 'M', 'F': 'F', 'Male': 'M', 'Female': 'F'}

        # Combine data from all datasets
        for i, (data_dnam, data_meta) in enumerate(self.datasets):
            if cpg_site not in data_dnam.index:
                print(f"Warning: CpG site '{cpg_site}' not found in dataset {i + 1}. Skipping this dataset.")
                continue
            methylation_data = data_dnam.loc[cpg_site]
            valid_samples = data_meta.index.intersection(methylation_data.index)
            plot_data = data_meta.loc[valid_samples, ['age', 'sex']].copy()
            plot_data['methylation'] = methylation_data[valid_samples]
            plot_data['sex'] = plot_data['sex'].replace(sex_mapping)
            plot_data['dataset'] = f'Dataset {i + 1}'
            combined_data.append(plot_data)

        combined_data = pd.concat(combined_data).dropna(subset=['age', 'methylation'])

        # Define bins for age groups with whole numbers
        min_age = int(np.floor(combined_data['age'].min()))
        max_age = int(np.ceil(combined_data['age'].max()))
        bins = np.linspace(min_age, max_age, age_bins + 1).astype(int)
        combined_data['age_group'] = pd.cut(
            combined_data['age'], 
            bins=bins, 
            include_lowest=True, 
            labels=[f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        )

        stats_summary = []

        # Group by age_group to compute statistics for each age range
        age_groups = combined_data['age_group'].unique()
        for age_group in age_groups:
            group_data = combined_data[combined_data['age_group'] == age_group]
            
            # Separate data by sex within the age group
            male_data = group_data[group_data['sex'] == 'M']['methylation']
            female_data = group_data[group_data['sex'] == 'F']['methylation']
            
            # Calculate statistics
            stats_row = {
                'age_group': age_group,
                'male_mean': male_data.mean(),
                'female_mean': female_data.mean(),
                'male_median': male_data.median(),
                'female_median': female_data.median(),
                'male_std': male_data.std(),
                'female_std': female_data.std(),
                'male_count': male_data.count(),
                'female_count': female_data.count()
            }

            # Perform t-test if both groups have more than 1 sample
            if len(male_data) > 1 and len(female_data) > 1:
                t_stat, p_val = stats.ttest_ind(male_data, female_data, nan_policy='omit')
                stats_row['p_value'] = p_val
                # Add significance stars based on p-value
                stats_row['significance'] = ''
                if p_val < 0.05:
                    stats_row['significance'] = '*'
                if p_val < 0.01:
                    stats_row['significance'] = '**'
                if p_val < 0.001:
                    stats_row['significance'] = '***'
            else:
                stats_row['p_value'] = None
                stats_row['significance'] = ''

            # Append statistics for the current age group
            stats_summary.append(stats_row)

        # Convert the summary list to a DataFrame for easy viewing
        stats_df = pd.DataFrame(stats_summary)
        return stats_df, combined_data

    def plot_methylation_by_age_sex_multiple(self, combined_data, cpg_site):
        """
        Visualizes methylation levels at a specific CpG site by age group and sex using combined data.

        Parameters:
        - combined_data: DataFrame with combined data from all datasets, including 'age_group' column.
        - cpg_site: CpG site ID to visualize.
        """
        plt.figure(figsize=(14, 8))
        sns.violinplot(x='age_group', y='methylation', hue='sex', data=combined_data, split=True, palette="Set2")
        plt.title(f'Methylation Level at {cpg_site} by Age Group and Sex Across Multiple Datasets')
        plt.xlabel('Age Group')
        plt.ylabel(f'Methylation Level at {cpg_site}')
        plt.legend(title="Sex")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def compute_methylation_vs_age_statistics(self, cpg_site, degree=2):
        """
        Computes regression statistics (linear and polynomial) for DNA methylation vs. age.

        Parameters:
        - cpg_site: CpG site ID to compute statistics for.
        - degree: Degree of the polynomial for the polynomial fit (default: 2).

        Returns:
        - combined_data: DataFrame with combined methylation and age data.
        - statistics: Dictionary containing linear and polynomial regression results.
        """
        combined_data = []

        for i, (data_dnam, data_meta) in enumerate(self.datasets):
            if cpg_site not in data_dnam.index:
                print(f"Warning: CpG site '{cpg_site}' not found in dataset {i + 1}. Skipping this dataset.")
                continue

            methylation_data = data_dnam.loc[cpg_site]
            valid_samples = data_meta.index.intersection(methylation_data.index)
            plot_data = data_meta.loc[valid_samples, ['age']].copy()
            plot_data['methylation'] = methylation_data[valid_samples]
            plot_data['dataset'] = f'Dataset {i + 1}'
            combined_data.append(plot_data)

        combined_data = pd.concat(combined_data).dropna(subset=['age', 'methylation'])
        
        ages = combined_data['age'].values.reshape(-1, 1)
        methylation_levels = combined_data['methylation'].values

        # Linear regression
        linear_model = LinearRegression()
        linear_model.fit(ages, methylation_levels)
        linear_pred = linear_model.predict(ages)
        r_squared_linear = r2_score(methylation_levels, linear_pred)
        corr, p_value = pearsonr(ages.flatten(), methylation_levels)

        # Polynomial regression
        poly = PolynomialFeatures(degree)
        ages_poly = poly.fit_transform(ages)
        poly_model = LinearRegression()
        poly_model.fit(ages_poly, methylation_levels)
        poly_pred = poly_model.predict(ages_poly)
        r_squared_poly = r2_score(methylation_levels, poly_pred)

        # Compile statistics
        statistics = {
            'linear_pred': linear_pred,
            'r_squared_linear': r_squared_linear,
            'corr': corr,
            'p_value': p_value,
            'poly_pred': poly_pred,
            'r_squared_poly': r_squared_poly,
            'degree': degree
        }

        # Print statistics
        print(f"Computed statistics for CpG site: {cpg_site}")
        print(f"Linear Regression R^2: {r_squared_linear:.4f}")
        print(f"Pearson Correlation (r): {corr:.4f}")
        print(f"Pearson Correlation p-value: {p_value:.4e}")
        print(f"Polynomial Regression (Degree {degree}) R^2: {r_squared_poly:.4f}")
        return combined_data, statistics

    def plot_methylation_vs_age(self, combined_data, statistics, cpg_site):
        """
        Plots the DNA methylation level of a specific CpG site against age,
        with both a linear and polynomial regression line.

        Parameters:
        - combined_data: DataFrame with combined methylation and age data.
        - statistics: Dictionary containing regression statistics.
        - cpg_site: CpG site ID to visualize.
        """
        ages = combined_data['age'].values
        methylation_levels = combined_data['methylation'].values

        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=ages, y=methylation_levels, hue=combined_data['dataset'], palette="viridis", s=10, alpha=0.6)

        # Linear fit line
        sns.lineplot(x=ages, y=statistics['linear_pred'], color='red', linestyle="--", label=f'Linear Fit (R^2 = {statistics["r_squared_linear"]:.2f})')

        # Polynomial fit line
        sns.lineplot(x=ages, y=statistics['poly_pred'], color='green', label=f'Polynomial Fit (Degree {statistics["degree"]}, R^2 = {statistics["r_squared_poly"]:.2f})')

        # Display statistics on the plot
        plt.text(0.05, 0.95, f'R^2 (Linear): {statistics["r_squared_linear"]:.2f}\nPearson r: {statistics["corr"]:.2f}\nP-value: {statistics["p_value"]:.3e}',
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, color='red')

        # Add title and labels
        plt.title(f"Methylation Level at {cpg_site}")
        plt.xlabel("Age")
        plt.ylabel("DNA Methylation")
        plt.ylim(0, 1)
        plt.xlim(ages.min(), ages.max())
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.show()









# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import ttest_ind, pearsonr
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score

# class DNAMethylationAnalyzer:
#     def __init__(self, metadata: pd.DataFrame, methylation_data: pd.DataFrame):
#         self.metadata = metadata
#         self.methylation_data = methylation_data

#     def preprocess_data(self):
#         """
#         Preprocesses the methylation data by handling missing values and ensuring data consistency.
#         """
#         # Drop rows with all NaN values (CpG sites with no data)
#         self.methylation_data.dropna(how='all', inplace=True)
#         # Fill NaN values with the mean value for each CpG site
#         self.methylation_data.fillna(self.methylation_data.mean(), inplace=True)

#     def identify_stable_cpg_sites(self, threshold=0.01):
#         """
#         Identifies stable CpG sites based on variance.
#         A CpG site is considered stable if its variance is below the given threshold.
        
#         :param threshold: Variance threshold to determine stability (default: 0.01)
#         :return: List of stable CpG site IDs
#         """
#         # Calculate variance for each CpG site (each row)
#         variances = self.methylation_data.var(axis=1)
#         # Identify CpG sites with variance below the threshold
#         stable_cpg_sites = variances[variances < threshold].index.tolist()        
#         return stable_cpg_sites

#     def identify_significant_cpg_sites(self, group_field: str = None, group1: str = None, group2: str = None, p_value_threshold=0.05):
#         """
#         Identifies significant CpG sites between two groups using a t-test.
        
#         :param group_field: The metadata field used to define groups (e.g., 'Disease State')
#         :param group1: The value of the group_field for group 1 (e.g., 'None')
#         :param group2: The value of the group_field for group 2 (e.g., 'COVID')
#         :param p_value_threshold: The p-value threshold for significance (default: 0.05)
#         :return: List of significant CpG site IDs
#         """
#         # Get sample IDs for each group
#         group1_samples = self.metadata[self.metadata[group_field] == group1].index.tolist()
#         group2_samples = self.metadata[self.metadata[group_field] == group2].index.tolist()

#         significant_cpg_sites = []

#         # Iterate over each CpG site and perform a t-test
#         for cpg_site, row in self.methylation_data.iterrows():
#             # Extract methylation values for the CpG site across group1 and group2 samples
#             group1_values = row[group1_samples].dropna()
#             group2_values = row[group2_samples].dropna()

#             # Perform t-test
#             if len(group1_values) > 1 and len(group2_values) > 1:  # Ensure enough data points
#                 t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
                
#                 # Check if p-value is below the threshold
#                 if p_value < p_value_threshold:
#                     significant_cpg_sites.append(cpg_site)

#         return significant_cpg_sites

#     def plot_cpg_against_age(self, cpg_site: str, degree: int = 2):
#         """
#         Plots the methylation level of a specific CpG site against age and allows for a specified polynomial degree.

#         :param cpg_site: The CpG site ID to plot
#         :param degree: The degree of the polynomial for the non-linear relationship (default: 1 for linear fit)
#         """

#         if cpg_site not in self.methylation_data.index:
#             print(f"CpG site {cpg_site} not found in the methylation data.")
#             return

#         # Create a DataFrame with methylation values and sample IDs
#         cpg_data = self.methylation_data.loc[cpg_site]
#         cpg_data_df = pd.DataFrame({'SampleID': cpg_data.index, 'Methylation': cpg_data.values})
#         merged_data = pd.merge(cpg_data_df, self.metadata, left_on='SampleID', right_index=True)
        
#         if 'age' not in merged_data.columns:
#             print("Age column not found in metadata.")
#             return

#         ages = merged_data['age'].values.reshape(-1, 1)
#         methylation_levels = merged_data['Methylation'].values

#         # Linear Fit
#         linear_model = LinearRegression()
#         linear_model.fit(ages, methylation_levels)
#         linear_pred = linear_model.predict(ages)
#         r_squared_linear = r2_score(methylation_levels, linear_pred)
#         corr, p_value = pearsonr(ages.flatten(), methylation_levels)

#         plt.figure(figsize=(12, 6))
#         sns.scatterplot(x=ages.flatten(), y=methylation_levels, color='blue', label='Data')
#         sns.lineplot(x=ages.flatten(), y=linear_pred, color='red', label='Linear Fit')
#         plt.xlabel('Age')
#         plt.ylabel('Methylation Level')
#         plt.title(f'CpG Site {cpg_site} Methylation vs Age (Linear Relationship)')
        
#         # Displaying statistics on the plot
#         plt.text(0.05, 0.95, f'R^2 (Linear): {r_squared_linear:.2f}\nPearson r: {corr:.2f}\nP-value: {p_value:.3e}',
#                  transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, color='red')
#         plt.legend()
#         plt.show()

#         # Non-Linear Relationship (Polynomial Fit)
#         poly = PolynomialFeatures(degree=degree)
#         ages_poly = poly.fit_transform(ages)
#         poly_model = LinearRegression()
#         poly_model.fit(ages_poly, methylation_levels)
#         poly_pred = poly_model.predict(ages_poly)
#         r_squared_poly = r2_score(methylation_levels, poly_pred)

#         plt.figure(figsize=(12, 6))
#         sns.scatterplot(x=ages.flatten(), y=methylation_levels, color='blue', label='Data')
#         sns.lineplot(x=ages.flatten(), y=poly_pred, color='green', label=f'Polynomial Fit (Degree {degree})')
#         plt.xlabel('Age')
#         plt.ylabel('Methylation Level')
#         plt.title(f'CpG Site {cpg_site} Methylation vs Age (Non-Linear Relationship)')

#         # Displaying polynomial statistics on the plot
#         plt.text(0.05, 0.95, f'R^2 (Polynomial Degree {degree}): {r_squared_poly:.2f}',
#                  transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, color='green')
#         plt.legend()
#         plt.show()
