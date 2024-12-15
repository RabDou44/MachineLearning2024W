import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import re


def visualise_results(results_df, metrics = 'mse',print_results=True, save = False, folder_name= "figure"):
    """Visualise the results of the model
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    results_df = results_df.copy()
    if metrics in results_df.columns:
        metrics = [metrics]
    else:
        metrics = results_df.columns[1:]
    results_df["model_name"] = results_df['model'].apply(lambda x: re.findall(r"(.*?)\(", x)[0])
    results_df["parameter"] = results_df['model'].apply(lambda x: re.findall(r"\((.*?)\=\d{1,}\)", x)[0])
    results_df["parameter_value"] = results_df['model'].apply(lambda x: re.findall(r"\=(\d{1,})", x)[0])

    for metric in metrics:

        # plot the performance of the model fro given metric e.g.: mse
        parameter_names = results_df["parameter"].unique()
        metric_results = results_df[["model_name", "parameter_value" , metric]]
        plt.figure(figsize=(10, 6))
        pname = parameter_names[0]

        # for pname in parameter_names:
        
        #     # for given parameter, plot the performance of the model e.g.: n_trees 
            
            
        #     # for model_name in metric_results["model_name"].unique():

        #     #     model_metric_results = metric_results[metric_results["model_name"] == model_name]
        #     #     sns.lineplot(data=model_metric_results, x="parameter_value",  y=metric, label=model_name)
            
        sns.lineplot(data=metric_results, x="parameter_value",  y=metric, hue='model_name')

        plt.xlabel(pname)
        plt.ylabel(metric)
        plt.title(f"Performance of {metric} over {pname}")

        # save fgures
        os.makedirs(f'./{folder_name}', exist_ok=True)
        if save:
            plt.savefig(f'./{folder_name}/{pname}_{metric}.png',)
        if print_results:
            plt.show()
        plt.close()
