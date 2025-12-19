import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim
from pytorch_forecasting import TimeSeriesDataSet, QuantileLoss
import ibmsalus.models.stratify as stratify
from ibmsalus.models.arlstm_model import ARLSTMModel
from ibmsalus.models.tft_model import TFTModel

from ibmsalus.models.tmds_model.tmds_data_utils import logging_percentage

import warnings
warnings.filterwarnings(action='ignore',module='pytorch_forecasting')

#The following code is to handle the unexpected error
#" dataloader RuntimeError: received 0 items of ancdata "
torch.multiprocessing.set_sharing_strategy('file_system')
if os.name == 'posix':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def model_func(dataset, **kwargs):
    """create the model from dataset
    """
    model_name =  kwargs.get('model_name', "tft")
    if model_name == "arlstm":
        model = ARLSTMModel(**kwargs)
        model.create_arlstm_model(dataset)
    else:
        model = TFTModel(**kwargs)
        model.create_tft_model(dataset)
        
    return model.model


def create_dataset(df, max_encoder_length, max_prediction_length,
                   time_idx='time_idx', target='count', group='group', prediction=False):
    """create training and testing dataset dataframe
    """
    if prediction:
        training_cutoff = df[time_idx].max()
    else:
        training_cutoff = df[time_idx].max() - max_prediction_length
    #training
    training = TimeSeriesDataSet(
        df[lambda x: x[time_idx] <= training_cutoff],
        time_idx= time_idx,  # column name of time of observation
        target= target,  # column name of target to predict
        group_ids=['group'],  # column name(s) for timeseries IDs
        max_encoder_length=max_encoder_length,  # how much history to use
        max_prediction_length=max_prediction_length,  # how far to predict into future
        # covariates static for a timeseries ID
        #static_categoricals=[ ... ],
        #static_reals=[ ... ],
        # covariates known and unknown in the future to inform prediction
        #time_varying_known_categoricals=[ ... ],
        time_varying_known_reals=[time_idx],
        #time_varying_unknown_categoricals=[ ... ],
        time_varying_unknown_reals=[target],
    )
    #validation
    if prediction:
        validation = None
    else:
        validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=2, stop_randomization=True)
    
    return training, validation


def model_forecast(model, val_dataloader):
    """running pytorch model and get the forecasting
    """
    outputs = []
    for (x, y) in val_dataloader:
        result = model(x)['prediction']
        if isinstance(result, list):
            cur_output = []
            for v in result:
                value = v.detach().numpy()
                cur_output.append(value[:, :, 3])
            cur_output = np.concatenate(cur_output)
            cur_output = np.swapaxes(cur_output, 0, 1)
            outputs.append(cur_output)
        else:
            output = result.detach().numpy()
            outputs.append(output[:, :, 3])
    outputs = np.concatenate(outputs)    
    return outputs


def conver_dataloader_numpy(val_dataloader):
    """get the input, groundtruth of validaiton data
    """
    val_x = []
    val_y = []
    for (x, y) in val_dataloader:
        val_x.append(x['encoder_target'].detach().numpy())
        val_y.append(y[0].detach().numpy())
    
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)
    return val_x, val_y


def save_checkpoint(state, filepath):
    """save the model weights
    """
    torch.save(state, filepath)
    
    
def load_model(model, checkpoint_path):
    """load the model checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = dict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    del checkpoint
    
    return model


def move_to(obj, device):
    """move the data into GPU
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, tuple):
        res = []
        for v in list(obj):
            res.append(move_to(v, device))
        return tuple(res)
    elif obj is None:
        return None
    else:
        print(obj)
        raise TypeError("Invalid type for move_to")
    
    
def create_optimizer(model, data_length, **kwargs):
    """create the model for training
    """
    learning_rate = kwargs.get('learning_rate', 0.002)
    epochs = kwargs.get('epochs', 40)
    opt = kwargs.get("optimizer", "Adam")
    anneal_epochs = kwargs.get("anneal_epochs", 5)

    if opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=2.5 * learning_rate, 
                                            steps_per_epoch=data_length,
                                            epochs=epochs,
                                            cycle_momentum=True,
                                            verbose=False)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_epochs=anneal_epochs, swa_lr=5*learning_rate)
    return optimizer, scheduler, swa_scheduler


def train_from_dataloader(model, optimizer, scheduler, loss_func, dataloader, device, lr_step=True):
    """Training the model from one dataloader
    """
    loss = 0
    for i, (x, y) in enumerate(dataloader):
        #to cude if needed
        x = move_to(x, device)
        y = move_to(y, device)
        optimizer.zero_grad()
        output = model(x)
        
        if isinstance(y[0], list):
            training_loss = []
            for j in range(len(y[0])):
                cur_loss = loss_func.loss(output['prediction'][i], y[0][i])
                training_loss.append(cur_loss)
            training_loss = torch.cat(training_loss, dim=0)
            training_loss = torch.sum(torch.mean(training_loss, dim=0))
        else:
            training_loss = loss_func.loss(output['prediction'], y[0])
            training_loss = torch.sum(torch.mean(training_loss, dim=0))
        
        training_loss.backward()
        #lr = scheduler.get_last_lr()[-1]
        loss += training_loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if lr_step:
            scheduler.step()

    return loss


def model_saving(model, epoch, swa_start, swa_epochs, **kwargs):
    """save the model
    """
    model_name = kwargs['model_name']
    model_path= kwargs['model_path']
    if swa_epochs > 0 and epoch > swa_start:
        swa_model_dict = model.state_dict()
        swa_model_dict = {k.replace('module.module','module'): v for k,v in swa_model_dict.items() if k != ""}
        swa_model_dict.pop('n_averaged', None)
        save_dict = {
            'arch': model_name,
            'state_dict': swa_model_dict,
        }
        name = f'{model_name}_swa_{epoch+1:02d}.pth.tar'
        save_checkpoint(save_dict, os.path.join(model_path, name))
    else:
        save_dict = {
            'arch': model_name,
            'state_dict': model.state_dict(),
        }
        name = f'{model_name}_{epoch+1:02d}.pth.tar'
        save_checkpoint(save_dict, os.path.join(model_path, name))


def train(model, model_file, training, logger, use_cuda=False, fine_tune=False, **kwargs):
    """loops to train the model
    """
    batch_size = kwargs.get('batch_size', 32)
    epochs = kwargs.get('epochs', 40)
    swa_epochs = kwargs.get('swa_epochs', 15)
    model_name =  kwargs.get('model_name', "tft")
    num_workers = kwargs.get("num_workers", 4)
    
    mtf_id = os.path.basename(os.path.dirname(model_file))
    target = os.path.basename(model_file).split('_')[0]
    
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if not use_cuda:
        num_workers = 0
        torch.set_num_threads(1)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    
    if fine_tune and os.path.exists(model_file):
        model = load_model(model, model_file)
        #possible freezing some layers
        epochs = 25
        swa_epochs = 15
        params = {
            "learning_rate": 0.001,
            "epochs": 30,
            "optimizer": "Adam",
            "anneal_epochs":5
        }
        optimizer, scheduler, swa_scheduler = create_optimizer(model, len(train_dataloader), **params)
        logger.info(f'Model finetune params:\n{json.dumps(params)}\n')
    else:
        optimizer, scheduler, swa_scheduler = create_optimizer(model, len(train_dataloader), **kwargs)
        logger.info(f'Model training params:\n{json.dumps(kwargs)}\n')
    
    #loss function
    loss_func = QuantileLoss()
    
    if use_cuda:
        model = torch.nn.DataParallel(model).to(device)
    model.train()
    
    swa_start = epochs - 1
    epochs = epochs + swa_epochs
    
    logger.info("Start the training ....")
    
    with tqdm(total=epochs) as t_bar:
        for epoch in range(epochs):
            lr_step = True
            if epoch > swa_start:
                lr_step = False
                
            loss = train_from_dataloader(model, optimizer, scheduler, loss_func, train_dataloader, device, lr_step=lr_step)
            loss /= len(train_dataloader)
                
            if epoch == swa_start:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            if epoch > swa_start:
                #print("Updating SWA model")
                swa_model.update_parameters(model)
                swa_scheduler.step()
            
            desc = f"Training {target} model on MTF {mtf_id}: epoch {epoch} loss: {loss:.06f}"
            t_bar.set_description(desc)
            t_bar.update(1)
            
            v = int(epoch / epochs * 100)
            logger.info(f"{desc}: {v}%")
            
    if swa_epochs > 0:
        swa_model_dict = swa_model.state_dict()
        swa_model_dict = {k.replace('module.module','module'): v for k,v in swa_model_dict.items() if k != ""}
        swa_model_dict.pop('n_averaged', None)
        save_dict = {
            'arch': model_name,
            'state_dict': swa_model_dict,
        }
        save_checkpoint(save_dict, model_file)
    else:
        save_dict = {
            'arch': model_name,
            'state_dict': model.state_dict(),
        }
        save_checkpoint(save_dict, model_file)
        
    logger.info(f"Taining is done and {model_file} is saved!\n")


def stratified_error(model_path, stratified_data):
    """stratified error analysis for each model output from 
    validation dataset
    """
    for k, data in stratified_data.items():
        if data:
            stratified_data[k] = np.expand_dims(np.concatenate(data), axis=-1)
    
    groundtruth = stratified_data['groundtruth']
    if groundtruth is None or len(groundtruth) == 0 or len(groundtruth.shape) < 2:
        return
    forecast_length = groundtruth.shape[1]
    
    model_dist = dict()
    for k, data in stratified_data.items():
        if k == 'groundtruth':
            continue
        if data is not None:
            model_dist[k] = stratify.calculate_distances(data, groundtruth)
        
    model_error = dict()
    max_error = 0
    for k, dist in model_dist.items():
        error = stratify.stratified_error(groundtruth, stratified_data[k], dist)
        model_error[k] = error
        max_error = max(max_error, np.max(error))
    
    #max_error = ((max_error//100) + 1) * 100
    max_error = int(max_error) + 5
    
    for k, error in model_error.items():
        plt.figure(num=1, clear=True)
        stratify.plot_error(error, forecast_length, max_error, k)
        plt.savefig(os.path.join(model_path, k + '_stratified_error.png'))