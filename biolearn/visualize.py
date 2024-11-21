import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from biolearn.model_gallery import ModelGallery

def visualize(model_name, plot_type, data_source, health_outcome_col=None):
    """
    Generate visualizations for a specified biological clock model and plot type.
    
    Parameters:
        model_name (str): Name of the biological clock model (e.g., "Horvath", "Hannum").
        plot_type (str): Type of plot to generate. Options:
                         - "health_outcome"
                         - "correlation_matrix"
                         - "hazard_ratios"
        data_source: Biolearn-compatible data source object (e.g., from DataLibrary).
        health_outcome_col (str, optional): Column name for health outcomes (required for "health_outcome" plot).
    
    Raises:
        ValueError: If an invalid plot type is provided.
    """
    # Initialize ModelGallery and get the requested model
    gallery = ModelGallery()
    model = gallery.get(model_name)
    
    # Predict using the selected model
    predictions = model.predict(data_source)
    data = pd.DataFrame({
        "Prediction": predictions["Predicted"],
        "Age": data_source.metadata.get("age", None)
    })
    
    if plot_type == "health_outcome":
        if health_outcome_col is None:
            raise ValueError("health_outcome_col is required for 'health_outcome' plot.")
        
        # Merge predictions with the health outcome data
        health_data = data_source.metadata[health_outcome_col]
        data["Health Outcome"] = health_data
        
        # Plot predictions vs health outcomes
        sns.scatterplot(data=data, x="Prediction", y="Health Outcome")
        plt.title(f"{model_name} Predictions vs {health_outcome_col}")
        plt.xlabel(f"{model_name} Prediction")
        plt.ylabel(health_outcome_col)
        plt.show()
    
    elif plot_type == "correlation_matrix":
        # Generate a correlation matrix for model predictions and other variables
        corr_data = data_source.metadata.copy()
        corr_data[model_name] = data["Prediction"]
        correlation_matrix = corr_data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Matrix: {model_name}")
        plt.show()
    
    elif plot_type == "hazard_ratios":
        # Generate hazard ratio plots (requires lifespan and healthspan columns in metadata)
        lifespan_col = "lifespan"  # Replace with actual lifespan column name
        healthspan_col = "healthspan"  # Replace with actual healthspan column name
        
        if lifespan_col not in data_source.metadata.columns or healthspan_col not in data_source.metadata.columns:
            raise ValueError(f"Metadata must contain '{lifespan_col}' and '{healthspan_col}' for hazard ratios.")
        
        lifespan = data_source.metadata[lifespan_col]
        healthspan = data_source.metadata[healthspan_col]
        
        data["Lifespan Hazard Ratio"] = (data["Prediction"] / lifespan).apply(np.log2)
        data["Healthspan Hazard Ratio"] = (data["Prediction"] / healthspan).apply(np.log)
        
        sns.scatterplot(data=data, x="Healthspan Hazard Ratio", y="Lifespan Hazard Ratio")
        plt.title(f"{model_name} Hazard Ratios")
        plt.xlabel("Healthspan Hazard Ratio (log scale)")
        plt.ylabel("Lifespan Hazard Ratio (log2-transformed)")
        plt.show()
    
    else:
        raise ValueError(f"Invalid plot type: {plot_type}. Choose from 'health_outcome', 'correlation_matrix', 'hazard_ratios'.")


from biolearn.data_library import DataLibrary

# Load a dataset (e.g., GSE41169)
data_source = DataLibrary().get("GSE41169")
data = data_source.load()

# Visualize clock predictions with health outcomes
visualize(model_name="Horvathv1", plot_type="health_outcome", data_source=data, health_outcome_col="mortality")

# Generate a correlation matrix
visualize(model_name="Horvathv1", plot_type="correlation_matrix", data_source=data)

# Plot hazard ratios
visualize(model_name="Horvathv1", plot_type="hazard_ratios", data_source=data)