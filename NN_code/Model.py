import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import  DataLoader, Dataset

import random
import os
import numpy as np
import csv


def _conv_stack(dilations, in_channels, out_channels, kernel_size):

    return nn.ModuleList(
        [
            nn.Conv1d(
                in_channels=(in_channels if i == 0 else out_channels),
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
                dtype=torch.float64,
            )
            for i, d in enumerate(dilations)
        ]
    )

class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        self.convs_sigm = _conv_stack(dilations, 1, num_channels, kernel_size)
        self.convs_tanh = _conv_stack(dilations, 1, num_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
            dtype=torch.float64,
        )

    def forward(self, x):
        out = x
        skips = []

        for conv_sigm, conv_tanh, residual in zip(
            self.convs_sigm, self.convs_tanh, self.residuals
        ):
            x = out
            out_sigm, out_tanh = conv_sigm(x), conv_tanh(x)
            # gated activation
            out = torch.tanh(out_tanh) * torch.sigmoid(out_sigm)
            skips.append(out)
            out = residual(out)
            out = out + x[:, :, -out.size(2) :]  # fit input with layer output

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out


def error_to_signal(y, y_pred):

    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)


def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


def csv_paths(csv_file):
    a = []
    delim = ''
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            a.append(delim.join(row))
    return a

def exlude_paths_containing(paths, spk1, spk2):
    b=[]
    if type(spk1) == type('s'):
        spk1=[spk1]
    if type(spk2) == type('s'):
        spk2=[spk2]
    spk=[]
    spk.extend(spk1)
    spk.extend(spk2)
    for path in paths:
        check = [1 for s in spk if path.find(s) != -1]
        if check.__len__() == 0:
            b.append(path)
    return b

def paths_containing(paths, spk):
    b=[]
    if type(spk) == type('s'):
        spk=[spk]
    for path in paths:
        check=[1 for s in spk if path.find(s) != -1]
        if check.__len__() != 0:
            b.append(path)

    return b

class CustomLoadDataset(Dataset):
    def __init__(self, data, W_len, Nstep): #delay
        self.data = data
        self.window_length = W_len
        self.Nstep = Nstep
        # self.delay = delay

    def __len__(self):
        return self.Nstep

    def __getitem__(self, idx):
        main_path = os.path.dirname(os.path.realpath(__file__))
        i = random.randrange(len(self.data))
        # i=idx
        path_x = self.data[i].split("")[1] #Name home directory
        path_x = main_path+ path_x
        path_x = path_x.split("_dbFS")[0]
        db = int(path_x.split("/")[-1])
        path_x = path_x+"_dbFS"
        path_y = path_x

        path_x = os.path.join(path_x, "x")
        path_x = path_x + self.data[i].split("_dbFS")[1]
        path_x = path_x + ".npy"

        path_y = os.path.join(path_y, "y")
        path_y = path_y + self.data[i].split("_dbFS")[1]
        path_y = path_y + ".npy"

        x = np.load(path_x)
        y = np.load(path_y)



        if len(y) < len(x):
            x = x[:len(y)]
        if len(y) > len(x):
            y = y[:len(x)]

        a = 0
        th = 5e-4
        n=0
        while a == 0:
            n=n+1
            if n>100:
                print("n>100")
            ii = random.randrange(len(y) - self.window_length)
            x_w = x[ii: ii + self.window_length].reshape((1, self.window_length))  # .reshape((1, self.window_length))
            if pow((x_w*pow(10, db/20)), 2).mean() > th:
                a = 1
        x = x_w
        y = y[ii: ii + self.window_length].reshape((1, self.window_length))
        return x,y


class PedalNet(L.LightningModule):
    def __init__(self, hparams):
        super(PedalNet, self).__init__()
        self.wavenet = WaveNet(
            num_channels=hparams["num_channels"],
            dilation_depth=hparams["dilation_depth"],
            num_repeat=hparams["num_repeat"],
            kernel_size=hparams["kernel_size"],

        )


        self.path_Dataset = hparams["path_Dataset"]
        self.N_workers = hparams["N_workers"]
        self.Data_val = hparams["Data_val"]
        self.Data_test = hparams["Data_test"]
        self.W_len = hparams["W_len"]
        self.N_step = hparams["N_step"]
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def prepare_data(self):
        # IMPORT PATHS FROM CSV
        paths = csv_paths(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),self.path_Dataset), 'data.csv'))
        paths_train = exlude_paths_containing(paths, self.Data_val, self.Data_test)
        paths_val = paths_containing(paths, self.Data_val)

        self.train_ds = CustomLoadDataset(data = paths_train, W_len=self.W_len, Nstep=self.N_step) #,delay=self.delay
        self.valid_ds = CustomLoadDataset(data = paths_val,  W_len=self.W_len, Nstep=self.N_step) #,delay=self.delay


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.wavenet.parameters(), lr=self.hparams.hparams["learning_rate"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.hparams["batch_size"],
            num_workers=self.N_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.hparams.hparams["batch_size"], num_workers=self.N_workers
        )

    def forward(self, x):
        return self.wavenet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        logs = {"loss": loss}
        self.training_step_outputs.append(loss)
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        self.validation_step_outputs.append(loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        logs = {"val_loss": epoch_average}
        self.log("validation_epoch_mean", epoch_average)
        self.validation_step_outputs.clear()
        return {"avg_val_loss": epoch_average, "log": logs}

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_mean", epoch_average)
        self.training_step_outputs.clear()