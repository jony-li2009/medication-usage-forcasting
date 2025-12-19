import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dtw import dtw
from copy import deepcopy

def calculate_distances(forecasts, actuals, mode='dtw'):
        distances = []
        for i in range(actuals.shape[0]):
            x = []
            for j in range(actuals.shape[2]):
                template = forecasts[i,:,j]
                query = actuals[i,:,j]
                
                if mode == 'dtw':
                    alignment = dtw(query, template)
                    x += [alignment.distance]
                else:
                    dist = np.sqrt(((template-query)**2).sum())
                    x += [dist]
                    
            distances += [x]
        distances = np.array(distances)
        return distances
    
def calculate_errors(actuals, forecasts, axis=1):
        matrix = np.absolute(actuals - forecasts).mean(axis = axis)
        return matrix
    
##final functions
def flattened_plot(array):
    finalflat = array.flatten()
    df = pd.DataFrame(finalflat, columns = ['distance'])
    hist = sns.histplot(data=df, x="distance")
    hist.set_xlabel("Distance")
    hist.set_ylabel("Count")
    hist.set(title='Histogram of Distances')
    return hist

def plot_scatter(x, y):
    scatter = sns.scatterplot(x.flatten(), y.flatten())
    scatter.set_xlabel("Distance")
    scatter.set_ylabel("Error")
    scatter.set(title='Distances vs. Errors')
    return scatter

def stratify_forecasts(actuals, forecasts, distances):
    
    """
    stratify forecast/actual pairs by profile
    """
    
    forecast_length = forecasts.shape[1]
    stratified = []

    # iterate over features
    for i in range(forecasts.shape[2]):
        subset_actuals = actuals[:,:,i]
        subset_forecasts = forecasts[:,:,i]
        subset_distances = distances[:,i]

        quants = [np.quantile(subset_distances,n/4) for n in range(1,5,1)]
        quantiles = []

        for j in range(len(quants)):
            if j == 0:
                mask = (subset_distances <= quants[j])
            else:
                mask = (subset_distances <= quants[j]) * (subset_distances > quants[j-1])
            quant_actuals = subset_actuals[mask]
            quant_forecasts = subset_forecasts[mask]
            quantiles += [(quant_actuals,quant_forecasts)]

        stratified += [quantiles]

    return stratified

def plot_stratified_error(actuals,forecasts,distances):
    
    forecast_length = actuals.shape[1]
    feature_errors = []

    for i in range(actuals.shape[2]):
        subset_actuals = actuals[:,:,i]
        subset_forecasts = forecasts[:,:,i]
        subset_distances = distances[:,i]

        quants = [np.quantile(subset_distances,n/4) for n in range(1,5,1)]
        quant_errors = []

        for j in range(len(quants)):
            if j == 0:
                mask = (subset_distances <= quants[j])
            else:
                mask = (subset_distances <= quants[j]) * (subset_distances > quants[j-1])
            quant_actuals = subset_actuals[mask]
            quant_forecasts = subset_forecasts[mask]
            quant_errors += [np.absolute(quant_actuals - quant_forecasts).mean(axis = 0)]

        feature_errors += [quant_errors]


    days = [i+1 for i in range(forecast_length)]
    for i, error in enumerate(feature_errors):
        error_df = pd.DataFrame({
            'day': days, 
            '1st Quartile': error[0],
            '2nd Quartile': error[1],
            '3rd Quartile': error[2],
            '4th Quartile': error[3]})

        #plt.figure(i)
        sns.lineplot(x='day', y='value', hue='variable', 
                     data=pd.melt(error_df, ['day']))
        #plt.ylim((0, 400))
        
        
def stratified_error(actuals, forecasts, distances):
    """calculate the stratified error
    """
    feature_errors = []

    for i in range(actuals.shape[2]):
        subset_actuals = actuals[:,:,i]
        subset_forecasts = forecasts[:,:,i]
        subset_distances = distances[:,i]

        quants = [np.quantile(subset_distances,n/4) for n in range(1,5,1)]
        quant_errors = []

        for j in range(len(quants)):
            if j == 0:
                mask = (subset_distances <= quants[j])
            else:
                mask = (subset_distances <= quants[j]) * (subset_distances > quants[j-1])
            quant_actuals = subset_actuals[mask]
            quant_forecasts = subset_forecasts[mask]
            quant_errors += [np.absolute(quant_actuals - quant_forecasts).mean(axis = 0)]

        feature_errors += [quant_errors]
        
    return feature_errors


def plot_error(feature_errors, forecast_length, max_error, title):
    """plot the error
    """
    days = [i+1 for i in range(forecast_length)]
    for i, error in enumerate(feature_errors):
        error_df = pd.DataFrame({
            'day': days, 
            '1st Quartile': error[0],
            '2nd Quartile': error[1],
            '3rd Quartile': error[2],
            '4th Quartile': error[3]})

        #plt.figure(i)
        sns.lineplot(x='day', y='value', hue='variable', 
                     data=pd.melt(error_df, ['day']))
        plt.ylim((0, max_error))
        plt.title(title)
        
