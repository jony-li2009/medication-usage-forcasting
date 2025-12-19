import os
from abc import ABC, abstractmethod
import torch
import torch.optim
from torch import nn
from pytorch_forecasting import TimeSeriesDataSet, QuantileLoss
from pytorch_forecasting.models.nn import LSTM
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModel
from typing import Any, Callable, Dict, List, Union
from pytorch_forecasting.utils import apply_to_list, to_list

import warnings
warnings.filterwarnings(action='ignore',module='pytorch_lightning')

class AutoRegressiveLSTMModel(AutoRegressiveBaseModel):
    def __init__(
        self,
        input_size: int,
        target: Union[str, List[str]],
        target_lags: Dict[str, Dict[str, int]],
        n_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        # arguments target and target_lags are required for autoregressive models
        # even though target_lags cannot be used without covariates
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)

        # use version of LSTM that can handle zero-length sequences
        self.lstm = LSTM(
            hidden_size=self.hparams.hidden_size,
            input_size=input_size,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        #self.output_layer = nn.Linear(self.hparams.hidden_size, len(kwargs['loss'].quantiles))
        self.multi_target = False
        if isinstance(target, str):  # single target
            self.output_layer = nn.Linear(self.hparams.hidden_size, len(self.loss.quantiles))
        else:  # multi target
            self.output_layer = nn.ModuleList(
                [nn.Linear(self.hparams.hidden_size, len(self.loss[0].quantiles)) for _ in target]
            )
            self.multi_target = True

    def encode(self, x: Dict[str, torch.Tensor]):
        # we need at least one encoding step as because the target needs to be lagged by one time step
        # because we use the custom LSTM, we do not have to require encoder lengths of > 1
        # but can handle lengths of >= 1
        assert x["encoder_lengths"].min() >= 1
        input_vector = x["encoder_cont"].clone()
        # lag target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        input_vector = input_vector[:, 1:]  # first time step cannot be used because of lagging

        # determine effective encoder_length length
        effective_encoder_lengths = x["encoder_lengths"] - 1
        # run through LSTM network
        _, hidden_state = self.lstm(
            input_vector, lengths=effective_encoder_lengths, enforce_sorted=False  # passing the lengths directly
        )  # second ouput is not needed (hidden state)
        return hidden_state

    def decode(self, x: Dict[str, torch.Tensor], hidden_state):
        # again lag target by one
        input_vector = x["decoder_cont"].clone()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        # but this time fill in missing target from encoder_cont at the first time step instead of throwing it away
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
            x["encoder_lengths"] - 1,
            self.target_positions.unsqueeze(-1),
        ].T
        input_vector[:, 0, self.target_positions] = last_encoder_target

        if self.training:  # training mode
            lstm_output, _ = self.lstm(input_vector, hidden_state, lengths=x["decoder_lengths"], enforce_sorted=False)

            # transform into right shape
            if self.multi_target: 
                prediction = [output_layer(lstm_output) for output_layer in self.output_layer]
            else:
                prediction = self.output_layer(lstm_output)

            prediction = self.transform_output(prediction, target_scale=x["target_scale"])

            # predictions are not yet rescaled
            return prediction

        else:  # prediction mode
            target_pos = self.target_positions

            def decode_one(idx, lagged_targets, hidden_state):
                x = input_vector[:, [idx]]
                # overwrite at target positions
                #x[:, 0, target_pos] = lagged_targets[-1]  # take most recent target (i.e. lag=1)
                #for lag, lag_positions in lagged_target_positions.items():
                #    if idx > lag:
                #        x[:, 0, lag_positions] = lagged_targets[-lag]

                lstm_output, hidden_state = self.lstm(x, hidden_state)
                # transform into right shape
                if self.multi_target:
                    prediction = [output_layer(lstm_output) for output_layer in self.output_layer]
                else:
                    prediction = self.output_layer(lstm_output)
                prediction = apply_to_list(prediction, lambda x: x[:, 0])  # select first time step
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=x["target_scale"],
                n_decoder_steps=input_vector.size(1),
            )
            # predictions are already rescaled
            return output

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden_state = self.encode(x)  # encode to hidden state
        output = self.decode(x, hidden_state)  # decode leveraging hidden state

        return self.to_network_output(prediction=output)

    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        first_hidden_state: Any,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Make predictions in auto-regressive manner.

        Supports only continuous targets.

        Args:
            decode_one (Callable): function that takes at least the following arguments:

                * ``idx`` (int): index of decoding step (from 0 to n_decoder_steps-1)
                * ``lagged_targets`` (List[torch.Tensor]): list of normalized targets.
                  List is ``idx + 1`` elements long with the most recent entry at the end, i.e.
                  ``previous_target = lagged_targets[-1]`` and in general ``lagged_targets[-lag]``.
                * ``hidden_state`` (Any): Current hidden state required for prediction.
                  Keys are variable names. Only lags that are greater than ``idx`` are included.
                * additional arguments are not dynamic but can be passed via the ``**kwargs`` argument

                And returns tuple of (not rescaled) network prediction output and hidden state for next
                auto-regressive step.

            first_target (Union[List[torch.Tensor], torch.Tensor]): first target value to use for decoding
            first_hidden_state (Any): first hidden state used for decoding
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale as in ``x``
            n_decoder_steps (int): number of decoding/prediction steps
            **kwargs: additional arguments that are passed to the decode_one function.

        Returns:
            Union[List[torch.Tensor], torch.Tensor]: re-scaled prediction

        """
        # make predictions which are fed into next step
        output = []
        current_target = first_target
        current_hidden_state = first_hidden_state

        normalized_output = [first_target]

        for idx in range(n_decoder_steps):
            # get lagged targets
            current_target, current_hidden_state = decode_one(
                idx, lagged_targets=normalized_output, hidden_state=current_hidden_state, **kwargs
            )

            # get prediction and its normalized version for the next step
            prediction, current_target = self.output_to_prediction(current_target, target_scale=target_scale)
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            output.append(prediction)
        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
        else:
            # for multi-targets
            num = len(self.hparams.target)
            outputs = [[] for _ in range(num)]
            for out in output:
                for idx in range(num):
                    outputs[idx].append(out[idx])
            output = [torch.stack(out, dim=1) for out in outputs]
        return output


class ARLSTMModel(ABC):
    """utility functions for AutoRegressive LSTM Model
    """
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get('hidden_size', 10) # Colin changed from 16 to 10
        self.hidden_continuous_size = kwargs.get('hidden_continuous_size', 16)
        self.learning_rate = kwargs.get('learning_rate', 0.002)
        #Added by Colin
        self.n_layers = kwargs.get('n_layers', 2)
        
    def create_arlstm_model(self, dataset):
        """create the arlstm model
        """
        input_size = len(dataset.time_varying_known_reals) + len(dataset.time_varying_unknown_reals)
        target = dataset.target
        if isinstance(target, str):
            loss = QuantileLoss()
        else:
            loss = [QuantileLoss() for _ in target]
        #create the model
        self.model = AutoRegressiveLSTMModel.from_dataset(
            # dataset
            dataset,
            # architecture hyperparameters
            input_size = input_size,
            hidden_size=self.hidden_size,
            #attention_head_size=1,
            dropout=0.1,
            #hidden_continuous_size=16, # Commented out by Colin
            # loss metric to optimize
            loss=loss,
            # logging frequency
            log_interval=2,
            # optimizer parameters
            learning_rate=self.learning_rate,
            reduce_on_plateau_patience=4,
            # Added by Colin
            n_layers=self.n_layers
        )

    