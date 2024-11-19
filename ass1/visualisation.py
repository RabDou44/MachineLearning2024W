import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

def visualise_results(resultds_df, folder_name, parameter_name, parmaeter_values, print_results=False):
    """Visualise the results of the model
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    results_cv = resultds_df[resultds_df['model'].str.endswith('_CV')]
    results_holdout = resultds_df[resultds_df['model'].str.endswith('_Holdout')]
    model_name = results_cv['model'].values[0].split('_')[0]

    for metric in ['accuracy', 'precision', 'recall', 'f1-score','timing']:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_cv, x=parmaeter_values ,  y=metric, label='CV')
        sns.lineplot(data=results_holdout, x=parmaeter_values,y=metric, label='Holdout')
        plt.xlabel(parameter_name)
        plt.ylabel(metric)

        # save fgures
        os.makedirs(f'./{folder_name}_figures/{model_name}', exist_ok=True)
        plt.savefig(f'./{folder_name}_figures/{model_name}/{parameter_name}_{metric}.png',)
        if print_results:
            plt.show()
        plt.close()

   
def plot_confusion_matrix(y_true, y_pred, folder_name):
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    os.makedirs("conf_matrix", exist_ok=True)
    plt.savefig(f'./conf_matrix/{folder_name}.png')
    plt.show()
    return plt