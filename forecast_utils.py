import os
import json
from random import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import shutil
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import ibmsalus.models.naive as naive
from ibmsalus.models.tft_model import TFTModel
from ibmsalus.models.var_model import VARModel
from ibmsalus.models.arlstm_model import ARLSTMModel
import ibmsalus.models.model_utils as model_utils


def dataframe_filtering(item_df):
    """remove the noise: date jump but the count keep same
    """
    date = item_df['date'].tolist()
    count = item_df['count'].tolist()
    m = 0
    while m < len(date) - 1:
        date_diff = (date[-1-m] - date[-2-m]).days
        count_diff = count[-1-m] - count[-2-m]
        if date_diff > 3 and count_diff == 0:
            m += 1
        else:
            break
    item_df.drop(item_df.tail(m).index, inplace = True)
    return item_df
    

def dataframe_preprocessing(item_df):
    """remove the duplicates and interpolation
    """
    item_df.replace([np.inf, -np.inf], 0, inplace=True)
    item_df['count'] =item_df['count'].fillna(0)
    #remove duplicate date by average
    item_df = item_df.groupby(['date'])['count'].apply(np.average).reset_index()
    #check the missing date
    date_range = pd.date_range(start=item_df.date.min(), end=item_df.date.max())
    item_df = item_df.set_index('date').reindex(date_range).rename_axis('date').reset_index()
    item_df['count'].interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    item_df['count'] = item_df['count'].astype(int)
    item_df['time_idx'] = [x.days for x in (item_df['date'] - item_df['date'][0])]
    return item_df


def item_model_train_func(item_csv_path, model_path, cached_path="", **kwargs):
    """training item model
    """
    #get the mtf and item type
    name = os.path.split(item_csv_path)[1]
    mtf, item_type = name.split('_')[:2]
    #read the dataframe, sort it and get needed columns
    df = pd.read_csv(os.path.join(item_csv_path, 'items.csv'))
    df['report_date'] = pd.to_datetime(df['report_date'])
    item_df = df.sort_values(by=['report_date']).reset_index(drop=True)
    #item_df['date'] = [x.days for x in item_df['report_date'] - item_df['report_date'][0]]
    item_df['date'] = item_df['report_date'].dt.date
    item_df = item_df.rename(columns={'on_hand': 'count'})
    #data filtering
    item_df = dataframe_filtering(item_df)
    #Now remove the duplicates and interpolation
    item_df = item_df[['date', 'count']]
    item_df = dataframe_preprocessing(item_df)
    
    cur_path = os.path.join(model_path, mtf + '_' + item_type)
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    
    #Get the training df and last data points for forecasting
    forecast_length = kwargs.get('forecast_length', 7)
    lookback_horizon = kwargs.get('lookback_horizon', 40)
    
    num_of_rows = item_df.shape[0]
    
    model_options = kwargs['model_options']
    
    min_num = forecast_length * 2 + lookback_horizon
    if model_options[2] < min_num:
        model_options[2] = min_num
        
    if num_of_rows <= 0:
        result = None   
    elif num_of_rows < model_options[0]:
        #persistence_model
        num = min(num_of_rows, lookback_horizon)
        df_forcasting = item_df.tail(num).reset_index(drop=True)
        result = {
            'mtf': mtf,
            'item_type': item_type,
            'model_type': 'persistence',
            'model_file': '',
            'df_forecasting': df_forcasting
        }
    elif num_of_rows < model_options[1]:
        #drift model
        num = min(num_of_rows, lookback_horizon)
        df_forcasting = item_df.tail(num).reset_index(drop=True)
        result = {
            'model_type': 'drift',
            'model_file': '',
            'df_forecasting': df_forcasting
        }
    elif num_of_rows < model_options[2]:
        #VAR/Autoregression
        num = min(num_of_rows, lookback_horizon)
        df_forcasting = item_df.tail(num).reset_index(drop=True)
        #training VAR and return model file
        item_df['count'] = item_df['count'] + np.random.rand(num_of_rows) * 5
        model = VARModel(**kwargs)
        model_file = f"var_{forecast_length}_{model.max_lag}.pkl"
        model_file = os.path.join(cur_path, model_file)
        df = item_df[['count', 'count']]
        model.train(df, model_file)
        
        result = {
            'model_type': 'var',
            'model_file': os.path.basename(model_file),
            'df_forecasting': df_forcasting
        }
    else:
        df_forcasting = item_df.tail(lookback_horizon).reset_index(drop=True)
        #convert the count into logscale if needed
        item_df['count'] = item_df['count'] + np.random.rand(num_of_rows) * 5
        logscale = kwargs.get('logscale', True)
        if logscale:
            item_df['count'] = np.log(item_df['count'] + 1)
        #add group information. If it has multiple time series segment, group them
        item_df['group'] = 1
        
        #create the training and validation dataset
        training, validation = model_utils.create_dataset(item_df, lookback_horizon, forecast_length)
        batch_size = kwargs.get('batch_size', 32)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        
        #training tft model
        model = TFTModel(**kwargs)
        model.create_tft_model(training)
        #create model name
        tft_model_file = f"tft_swa_{forecast_length}_{lookback_horizon}.pth.tar"
        tft_model_file = os.path.join(cur_path, tft_model_file)
        model_utils.train(model.model, tft_model_file, training, **kwargs)
        #evaludate the model
        tft_output = model_utils.model_forecast(model.model, val_dataloader)
        
        #traing arlstm model
        model = ARLSTMModel(**kwargs)
        model.create_arlstm_model(training)
        #create model name
        arlstm_model_file = f"arlstm_swa_{forecast_length}_{lookback_horizon}.pth.tar"
        arlstm_model_file = os.path.join(cur_path, arlstm_model_file)
        model_utils.train(model.model, arlstm_model_file, training, **kwargs)
        arlstm_output = model_utils.model_forecast(model.model, val_dataloader)
            
        model_file = [os.path.basename(tft_model_file), os.path.basename(arlstm_model_file)]
        
        #evaluate the naive model and get the groundtruth
        val_x, val_y = model_utils.conver_dataloader_numpy(val_dataloader)
        if logscale:
            val_x = np.exp(val_x).astype(int) - 1
            val_y = np.exp(val_y).astype(int) - 1
            tft_output = np.exp(tft_output).astype(int) - 1
            arlstm_output = np.exp(arlstm_output).astype(int) - 1
        
        val_x[val_x < 1] = 1
        
        val_x = np.expand_dims(val_x, axis=-1)
        val_y = np.expand_dims(val_y, axis=-1)
        
        drift = naive.DriftModel()
        drift.history = val_x
        drift.forecast_length=forecast_length
        drift.fit()
        drift_output = np.squeeze(drift.forecast, axis=-1)
        
        persistence = naive.PersistenceModel()
        persistence.history = val_x
        persistence.forecast_length=forecast_length
        persistence.fit()
        persistence_output = np.squeeze(persistence.forecast, axis=-1)
        
        val_y = np.squeeze(val_y, axis=-1)
        
        result = {
            'model_type': 'deeplearning',
            'model_file': model_file,
            'df_forecasting': df_forcasting,
            'tft_output': tft_output,
            'arlstm_output': arlstm_output,
            'drift_output': drift_output,
            'persistence_output':persistence_output,
            'groundtruth': val_y
        }
        
    #reset the time_idx for df_forecasting
    if result is not None:
        #save the df for forecasting
        df = result['df_forecasting']
        m = df['time_idx'].iloc[0]
        df['time_idx'] -= m
        #now save the data in each folder
        df['mtf_id'] = mtf
        df['item_type'] = item_type
        filename = os.path.join(cur_path, 'df_forecasting.csv')
        df.to_csv(filename, index=False)
        #save the mode dict
        model_dict = {
            'model_type': result["model_type"],
            'model_file': result["model_file"]
        }
        forecast_model_dict = {
            'params': kwargs,
            'models':model_dict,
        }
        filename = os.path.join(cur_path, 'forecast_model_dict.json')
        with open(filename, 'w') as f:
            json.dump(forecast_model_dict, f, indent=4)
            
        #save the copy of model into cached folder
        if len(cached_path) > 0 and os.path.exists(cached_path):
            dst_path = os.path.join(cached_path, mtf + '_' + item_type )
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            os.makedirs(dst_path)
            #backup models
            names = os.listdir(cur_path)
            for name in names:
                if name.endswith('.csv') and name != "df_forecasting.csv":
                    continue
                if name.endswith('.png'):
                    continue
                filename = os.path.join(cur_path, name)
                if os.path.isdir(filename):
                    continue
                shutil.copy(filename, dst_path)
            
    return result


def item_model_train(item_csv_path, model_path, cached_path="", **kwargs):
    """training the model
    """
    try:
        return item_model_train_func(item_csv_path, model_path, cached_path, **kwargs)
    except Exception as e:
        print(e)
        return None


def plot_forecasting(historical, forecast, path, y_truth=None):
    """draw the forcasting comparing the groundtruth
    """
    t = [i - historical.shape[0] for i in range(historical.shape[0])]
    fig = plt.figure(num=1, clear=True)
    #fig.clear()
    plt.plot(t, historical, 'b', label='x')
    t = list(range(forecast.shape[0]))
    if y_truth:
        plt.plot(t, y_truth, 'r', label='groundtruth')
    plt.plot(t, forecast, 'g-x', label='forecast')
    plt.legend()
    plt.savefig(os.path.join(path, 'forecasting.png'))
    #plt.close(fig)


def item_model_predict_func(item_path):
    """training item model
    """
    filename = os.path.join(item_path, 'df_forecasting.csv')
    if not os.path.exists(filename):
        return None
    item_df = pd.read_csv(filename)
    
    filename = os.path.join(item_path, 'forecast_model_dict.json')
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        model_dict = json.load(f)
    model_params = model_dict['params']
    model_info = model_dict['models']
    model_type = model_info["model_type"]
    model_file = model_info["model_file"]
    
    cur_path = item_path
    if model_type == 'deeplearning':
        #choose tft only right now
        model_path =os.path.join(cur_path, model_file[0])
        if not os.path.exists(model_path):
            return None
    elif model_type =='var':
        model_path =os.path.join(cur_path, model_file)
        if not os.path.exists(model_path):
            return None
    else:
        model_path = cur_path
    
    name = os.path.basename(cur_path)
    mtf, item_type = name.split('_')[:2]
    #print(f'model prediction for item {item_type} in MTF {mtf} ...')
    
    lookback_horizon = model_params['lookback_horizon']
    prediction_length = model_params['forecast_length']
    logscale = model_params['logscale']
    
    forecast_date = item_df['date'].iat[-1]
    
    if model_type == 'persistence':
        historical = item_df['count'].tolist()
        historical = np.array(historical)
        val_x = np.array([historical])
        val_x = np.expand_dims(val_x, axis=-1)
        model = naive.PersistenceModel()
        model.history = val_x
        model.forecast_length = prediction_length
        model.fit()
        forecasting= model.forecast[0, :, 0]
    elif model_type == 'drift':
        historical = item_df['count'].tolist()
        historical = np.array(historical)
        val_x = np.array([historical])
        val_x = np.expand_dims(val_x, axis=-1)
        model = naive.DriftModel()
        model.history = val_x
        model.forecast_length = prediction_length
        model.fit()
        forecasting= model.forecast[0, :, 0]
    elif model_type == 'var':
        model = VARModel(**model_params)
        model.load_model(model_path)
        historical = item_df['count'].tolist()
        historical = np.array(historical)
        item_df['count'] = item_df['count'] + np.random.rand(item_df.shape[0]) * 5
        df = item_df[['count', 'count']]
        forecasting = model.predict(df)[:, 0].astype(int)
        model_path = os.path.dirname(model_path)
    else:
        item_df['count'] = item_df['count'] + np.random.rand(item_df.shape[0]) * 5
        if logscale:
            item_df['count'] = np.log(item_df['count'] + 1)
        model = TFTModel(**model_params)
        #add row with count 0
        for _ in range(prediction_length):
            row = {'time_idx': lookback_horizon, 'count': 0}
            #item_df = item_df.append(row, ignore_index=True)
            item_df.loc[lookback_horizon] = row
            lookback_horizon += 1
        item_df['time_idx'] = item_df['time_idx'].astype(int)
            
        item_df['group'] = 1
        dataset, _ = model_utils.create_dataset(item_df, lookback_horizon - prediction_length, prediction_length, prediction=True)
        model.create_tft_model(dataset)
        model_utils.load_model(model.model, model_path)
        dataloader = dataset.to_dataloader(train=False, batch_size=1)
        #run the model
        for i, (x, y) in enumerate(dataloader):
            forecasting = model.model(x)['prediction'].detach().numpy()
            forecasting[forecasting < 1] = 0
        
            historical = x['encoder_target'].detach().numpy().squeeze()
            #groudtruth = y[0].detach().numpy().squeeze()
            forecasting = forecasting[0, :, 3]
        
        if logscale:
            forecasting = np.exp(forecasting).astype(int) - 1
            historical = np.exp(historical).astype(int) - 1
        model_path = os.path.dirname(model_path)
    
    forecasting[forecasting < 0] = 0
    plot_forecasting(historical, forecasting, model_path)
    
    #calculate the trend
    cur_value = historical[-1]
    num = min(prediction_length, historical.shape[0])
    data = historical[:num]
    if data.shape[0] == 1:
        average = data[0]
        cur_trend = 0
    else:
        average = np.average(data)
        t = np.arange(num).reshape(-1, 1)
        data = data.reshape(-1, 1)
        reg = LinearRegression().fit(t, data)
        cur_trend = reg.coef_[0][0]
        
    #calcuate the forecasting trend
    pred = np.concatenate([[cur_value], forecasting])
    t = np.arange(pred.shape[0]).reshape(-1, 1)
    pred = pred.reshape(-1, 1)
    reg = LinearRegression().fit(t, pred)
    pred_trend = reg.coef_[0][0]
    
    forecasting = forecasting.tolist()
    
    result = {
        **model_info,
        'forecast_date': forecast_date,
        'forecast': forecasting,
        'current_value': float(cur_value),
        'average': float(average),
        'current_trend': float(cur_trend),
        'predict_trend': float(pred_trend)
    }
    
    return mtf, item_type, result


def item_model_predict(item_path):
    """training item model
    """
    try:
        return item_model_predict_func(item_path)
    except Exception as e:
        print(e)
        return None
    

def model_train(item_csv_list, **kwargs):
    """Given the list of csv file folder, train the model for each item in each MTF.
    """
    model_path = kwargs.get('model_path', os.getcwd())
    cached_path = kwargs.get('cached_path',"")
    num_process = kwargs.get('num_process', 4)
    
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model_params = {
        'model_options': kwargs['model_options'],
        'lookback_horizon': kwargs['lookback_horizon'],
        'forecast_length': kwargs['forecast_length'],
        'logscale': kwargs['logscale']
    }
    
    stratified_data = {
        "tft_model": [],
        "arlstm_model": [],
        "drift_model": [],
        "persistence_model": [],
        "groundtruth": []
    }
    
    with mp.Pool(processes=num_process) as pool:
        for result in pool.imap_unordered(partial(
                item_model_train, model_path=model_path, cached_path=cached_path, **model_params), item_csv_list):
            if result is None:
                continue
            model_type = result['model_type']
            if model_type == 'deeplearning':
                stratified_data['tft_model'].append(result['tft_output'])
                stratified_data['arlstm_model'].append(result['arlstm_output'])
                stratified_data['drift_model'].append(result['drift_output'])
                stratified_data['persistence_model'].append(result['persistence_output'])
                stratified_data['groundtruth'].append(result['groundtruth'])
    
    return stratified_data
    
    
def model_predict(item_csv_list, num_process=4):
    """model forecasting
    """
    model_dict = dict()
    with mp.Pool(processes=num_process) as pool:
        for m in pool.imap(item_model_predict, item_csv_list):
            if m is None:
                continue
            mtf, item_type, result = m
            if mtf not in model_dict:
                model_dict[mtf] = dict()
            model_dict[mtf][item_type] = result
    
    return model_dict


def mtf_item_model_run(mtf_path, **kwargs):
    """Given the list of csv file folder, train the model for each item in each MTF.
    """
    name_col = "name"
    quantity_col = "quantity"
    quantity_type_col = "quantity_type"
        
    time_series_name = kwargs.get('time_series_file_name', 'code_meds_time_series.csv')
    df = pd.read_csv(os.path.join(mtf_path, time_series_name))
    df = df.dropna().reset_index(drop=True)
    df = df[df[quantity_type_col]=="on_hand"].reset_index(drop=True)
    item_names = df[name_col].unique()
    
    #get the dataframe list for each item
    item_dfs = []
    for name in item_names:
        item_df = df[df[name_col]==name].reset_index(drop=True)
        date_range = pd.date_range(start=item_df.date.min(), end=item_df.date.max())
        item_df = item_df.set_index('date').reindex(date_range).rename_axis('date').reset_index()
        item_df[quantity_col].interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
        item_df[quantity_col] = item_df[quantity_col].astype(int)
        item_df['time_idx'] = [x.days for x in (item_df['date'] - item_df['date'][0])]
        item_dfs.append(item_df)
    
    num_process = kwargs.get('num_process', 4)
    model_params = {
        'model_options': kwargs['model_options'],
        'lookback_horizon': kwargs['lookback_horizon'],
        'forecast_length': kwargs['forecast_length'],
        'logscale': kwargs['logscale']
    }
    
    stratified_data = {
        "tft_model": [],
        "arlstm_model": [],
        "drift_model": [],
        "persistence_model": [],
        "groundtruth": []
    }
    
    # with mp.Pool(processes=num_process) as pool:
    #     for result in pool.imap_unordered(partial(
    #             item_model_train, model_path=model_path, cached_path=cached_path, **model_params), item_csv_list):
    #         if result is None:
    #             continue
    #         model_type = result['model_type']
    #         if model_type == 'deeplearning':
    #             stratified_data['tft_model'].append(result['tft_output'])
    #             stratified_data['arlstm_model'].append(result['arlstm_output'])
    #             stratified_data['drift_model'].append(result['drift_output'])
    #             stratified_data['persistence_model'].append(result['persistence_output'])
    #             stratified_data['groundtruth'].append(result['groundtruth'])
    
    return stratified_data