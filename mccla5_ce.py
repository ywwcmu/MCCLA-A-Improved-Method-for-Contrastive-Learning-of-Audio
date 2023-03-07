import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
import random
import librosa
import numpy as np
import os
from glob import glob
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# from audio_encoder.audio_processing import random_crop, random_mask, random_multiply,pre_process_audio_mel_t
# from audio_encoder.encoder import Cola

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 0

input_length = 16000 * 30
n_mels = 64
def pre_process_audio_mel_t(audio, sample_rate=16000):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40

    return mel_db.T

def random_crop(data, crop_size=128):
    start = int(random.random() * (data.shape[0] - crop_size))
    # print(start)
    return data[start : (start + crop_size), :]

def random_mask(data, rate_start=0.1, rate_seq=0.2):
    new_data = data.copy()
    mean = new_data.mean()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if random.random() < rate_start or (
            prev_zero and random.random() < rate_seq
        ):
            prev_zero = True
            new_data[i, :] = mean
        else:
            prev_zero = False

    return new_data

def random_multiply(data):
    new_data = data.copy()
    return new_data * (0.9 + random.random() / 5.)

class Encoder(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(Encoder, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn1(x)
        x = self.efficientnet(x)

        y = x.squeeze(3).squeeze(2)

        return y


class Cola(pl.LightningModule):
    def __init__(self, p=0.1, n=2):
        super().__init__()
        self.save_hyperparameters()

        self.p = p

        self.do = torch.nn.Dropout(p=self.p)

        self.encoder = Encoder(drop_connect_rate=p)

        self.g = torch.nn.Linear(1280, 512)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=512)
        self.linear = torch.nn.Linear(512, 512, bias=False)
        self.n=n

    def forward(self, x):
        x1, x2, x3, x4, x5= x

        x1 = self.do(self.encoder(x1))
        x1 = self.do(self.g(x1))
        x1 = self.do(torch.tanh(self.layer_norm(x1)))

        x2 = self.do(self.encoder(x2))
        x2 = self.do(self.g(x2))
        x2 = self.do(torch.tanh(self.layer_norm(x2)))

        x3 = self.do(self.encoder(x3))
        x3 = self.do(self.g(x3))
        x3 = self.do(torch.tanh(self.layer_norm(x3)))

        x4 = self.do(self.encoder(x4))
        x4 = self.do(self.g(x4))
        x4 = self.do(torch.tanh(self.layer_norm(x4)))

        x5 = self.do(self.encoder(x4))
        x5 = self.do(self.g(x4))
        x5 = self.do(torch.tanh(self.layer_norm(x4)))

        x11 = self.linear(x1)
        x22 = self.linear(x2)
        x33 = self.linear(x3)
        x44 = self.linear(x4)
        x55 = self.linear(x5)

        return x1, x2, x3, x4, x5, x11, x22, x33, x44, x55


    def training_step(self, x, batch_idx):
        x1, x2, x3, x4, x5, x11, x22, x33, x44, x55 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat1 = torch.mm(x11, x2.t())
        y_hat2 = torch.mm(x22, x3.t())
        y_hat3 = torch.mm(x33, x4.t())
        y_hat4 = torch.mm(x44, x5.t())
        y_hat5 = torch.mm(x11, x3.t())
        y_hat6 = torch.mm(x11, x4.t())
        y_hat7 = torch.mm(x11, x5.t())
        y_hat8 = torch.mm(x22, x4.t())
        y_hat9 = torch.mm(x33, x5.t())
        y_hat10 = torch.mm(x44, x5.t())

        diag1 = torch.diag(y_hat1)
        diag2 = torch.diag(y_hat2)
        diag3 = torch.diag(y_hat3)
        diag4 = torch.diag(y_hat4)
        diag5 = torch.diag(y_hat5)
        diag6 = torch.diag(y_hat6)
        diag7 = torch.diag(y_hat7)
        diag8 = torch.diag(y_hat8)
        diag9 = torch.diag(y_hat9)
        diag10 = torch.diag(y_hat10)

        diag1 = torch.minimum(torch.minimum(torch.minimum(diag1,diag2),torch.minimum(diag3,diag4)),diag5)
        diag2 = torch.minimum(torch.minimum(torch.minimum(diag6,diag7),torch.minimum(diag8,diag9)),diag10)
        diag=torch.minimum(diag1,diag2)

        y_hata = torch.maximum(torch.maximum(torch.maximum(y_hat1,y_hat2),torch.maximum(y_hat3,y_hat4)),y_hat5)
        y_hatb = torch.maximum(torch.maximum(torch.maximum(y_hat6,y_hat7),torch.maximum(y_hat8,y_hat9)),y_hat10)
        y_hat=torch.maximum(y_hata,y_hatb)

        y_hat_new = torch.sub(y_hat,torch.diag_embed(torch.diag(y_hat)))
        y_hat_new = torch.add(y_hat_new,torch.diag_embed(diag))

        loss = F.cross_entropy(y_hat_new, y)

        _, predicted1 = torch.max(y_hat1, 1)
        _, predicted2 = torch.max(y_hat2, 1)
        _, predicted3 = torch.max(y_hat3, 1)
        _, predicted4 = torch.max(y_hat4, 1)
        _, predicted5 = torch.max(y_hat5, 1)
        _, predicted6 = torch.max(y_hat6, 1)
        _, predicted7 = torch.max(y_hat7, 1)
        _, predicted8 = torch.max(y_hat8, 1)
        _, predicted9 = torch.max(y_hat9, 1)
        _, predicted10 = torch.max(y_hat10, 1)

        acc1 = (predicted1 == y).double().mean()
        acc2 = (predicted2 == y).double().mean()
        acc3 = (predicted3 == y).double().mean()
        acc4 = (predicted4 == y).double().mean()
        acc5 = (predicted5 == y).double().mean()
        acc6 = (predicted6 == y).double().mean()
        acc7 = (predicted7 == y).double().mean()
        acc8 = (predicted8 == y).double().mean()
        acc9 = (predicted9 == y).double().mean()
        acc10 = (predicted10 == y).double().mean()

        acc = (acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8+acc9+acc10)/10

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, x, batch_idx):
        x1, x2, x3, x4, x5, x11, x22, x33, x44, x55 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat1 = torch.mm(x11, x2.t())
        y_hat2 = torch.mm(x22, x3.t())
        y_hat3 = torch.mm(x33, x4.t())
        y_hat4 = torch.mm(x44, x5.t())
        y_hat5 = torch.mm(x11, x3.t())
        y_hat6 = torch.mm(x11, x4.t())
        y_hat7 = torch.mm(x11, x5.t())
        y_hat8 = torch.mm(x22, x4.t())
        y_hat9 = torch.mm(x33, x5.t())
        y_hat10 = torch.mm(x44, x5.t())

        diag1 = torch.diag(y_hat1)
        diag2 = torch.diag(y_hat2)
        diag3 = torch.diag(y_hat3)
        diag4 = torch.diag(y_hat4)
        diag5 = torch.diag(y_hat5)
        diag6 = torch.diag(y_hat6)
        diag7 = torch.diag(y_hat7)
        diag8 = torch.diag(y_hat8)
        diag9 = torch.diag(y_hat9)
        diag10 = torch.diag(y_hat10)

        diag1 = torch.minimum(torch.minimum(torch.minimum(diag1,diag2),torch.minimum(diag3,diag4)),diag5)
        diag2 = torch.minimum(torch.minimum(torch.minimum(diag6,diag7),torch.minimum(diag8,diag9)),diag10)
        diag=torch.minimum(diag1,diag2)

        y_hata = torch.maximum(torch.maximum(torch.maximum(y_hat1,y_hat2),torch.maximum(y_hat3,y_hat4)),y_hat5)
        y_hatb = torch.maximum(torch.maximum(torch.maximum(y_hat6,y_hat7),torch.maximum(y_hat8,y_hat9)),y_hat10)
        y_hat=torch.maximum(y_hata,y_hatb)

        y_hat_new = torch.sub(y_hat,torch.diag_embed(torch.diag(y_hat)))
        y_hat_new = torch.add(y_hat_new,torch.diag_embed(diag))

        loss = F.cross_entropy(y_hat_new, y)

        _, predicted1 = torch.max(y_hat1, 1)
        _, predicted2 = torch.max(y_hat2, 1)
        _, predicted3 = torch.max(y_hat3, 1)
        _, predicted4 = torch.max(y_hat4, 1)
        _, predicted5 = torch.max(y_hat5, 1)
        _, predicted6 = torch.max(y_hat6, 1)
        _, predicted7 = torch.max(y_hat7, 1)
        _, predicted8 = torch.max(y_hat8, 1)
        _, predicted9 = torch.max(y_hat9, 1)
        _, predicted10 = torch.max(y_hat10, 1)

        acc1 = (predicted1 == y).double().mean()
        acc2 = (predicted2 == y).double().mean()
        acc3 = (predicted3 == y).double().mean()
        acc4 = (predicted4 == y).double().mean()
        acc5 = (predicted5 == y).double().mean()
        acc6 = (predicted6 == y).double().mean()
        acc7 = (predicted7 == y).double().mean()
        acc8 = (predicted8 == y).double().mean()
        acc9 = (predicted9 == y).double().mean()
        acc10 = (predicted10 == y).double().mean()

        acc = (acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8+acc9+acc10)/10

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def test_step(self, x, batch_idx):
        x1, x2, x3, x4, x5, x11, x22, x33, x44, x55 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat1 = torch.mm(x11, x2.t())
        y_hat2 = torch.mm(x22, x3.t())
        y_hat3 = torch.mm(x33, x4.t())
        y_hat4 = torch.mm(x44, x5.t())
        y_hat5 = torch.mm(x11, x3.t())
        y_hat6 = torch.mm(x11, x4.t())
        y_hat7 = torch.mm(x11, x5.t())
        y_hat8 = torch.mm(x22, x4.t())
        y_hat9 = torch.mm(x33, x5.t())
        y_hat10 = torch.mm(x44, x5.t())

        diag1 = torch.diag(y_hat1)
        diag2 = torch.diag(y_hat2)
        diag3 = torch.diag(y_hat3)
        diag4 = torch.diag(y_hat4)
        diag5 = torch.diag(y_hat5)
        diag6 = torch.diag(y_hat6)
        diag7 = torch.diag(y_hat7)
        diag8 = torch.diag(y_hat8)
        diag9 = torch.diag(y_hat9)
        diag10 = torch.diag(y_hat10)

        diag1 = torch.minimum(torch.minimum(torch.minimum(diag1,diag2),torch.minimum(diag3,diag4)),diag5)
        diag2 = torch.minimum(torch.minimum(torch.minimum(diag6,diag7),torch.minimum(diag8,diag9)),diag10)
        diag=torch.minimum(diag1,diag2)

        y_hata = torch.maximum(torch.maximum(torch.maximum(y_hat1,y_hat2),torch.maximum(y_hat3,y_hat4)),y_hat5)
        y_hatb = torch.maximum(torch.maximum(torch.maximum(y_hat6,y_hat7),torch.maximum(y_hat8,y_hat9)),y_hat10)
        y_hat=torch.maximum(y_hata,y_hatb)

        y_hat_new = torch.sub(y_hat,torch.diag_embed(torch.diag(y_hat)))
        y_hat_new = torch.add(y_hat_new,torch.diag_embed(diag))

        loss = F.cross_entropy(y_hat_new, y)

        _, predicted1 = torch.max(y_hat1, 1)
        _, predicted2 = torch.max(y_hat2, 1)
        _, predicted3 = torch.max(y_hat3, 1)
        _, predicted4 = torch.max(y_hat4, 1)
        _, predicted5 = torch.max(y_hat5, 1)
        _, predicted6 = torch.max(y_hat6, 1)
        _, predicted7 = torch.max(y_hat7, 1)
        _, predicted8 = torch.max(y_hat8, 1)
        _, predicted9 = torch.max(y_hat9, 1)
        _, predicted10 = torch.max(y_hat10, 1)

        acc1 = (predicted1 == y).double().mean()
        acc2 = (predicted2 == y).double().mean()
        acc3 = (predicted3 == y).double().mean()
        acc4 = (predicted4 == y).double().mean()
        acc5 = (predicted5 == y).double().mean()
        acc6 = (predicted6 == y).double().mean()
        acc7 = (predicted7 == y).double().mean()
        acc8 = (predicted8 == y).double().mean()
        acc9 = (predicted9 == y).double().mean()
        acc10 = (predicted10 == y).double().mean()

        acc = (acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8+acc9+acc10)/10

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=100, augment=True):
        self.data = data
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = self.data[idx]

        # x = np.load(npy_path, encoding='latin1')
        x = np.load(npy_path, allow_pickle=True)
        # print(x.shape)
        #x = pre_process_audio_mel_t(x)
        x=np.transpose(x)
        # print(x.shape)
        if self.augment:
            x = random_mask(x)

        if x.shape[0]<100:
          x = np.zeros((100,128))

        x1 = random_crop(x, crop_size=self.max_len)
        x2 = random_crop(x, crop_size=self.max_len)
        x3 = random_crop(x, crop_size=self.max_len)
        x4 = random_crop(x, crop_size=self.max_len)
        x5 = random_crop(x, crop_size=self.max_len)
        # print(x1.shape)
        # print(x2.shape)

        if self.augment:
            x1 = random_multiply(x1)
            x2 = random_multiply(x2)
            x3 = random_multiply(x3)
            x4 = random_multiply(x4)
            x5 = random_multiply(x5)

        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        x3 = torch.tensor(x3, dtype=torch.float)
        x4 = torch.tensor(x4, dtype=torch.float)
        x5 = torch.tensor(x5, dtype=torch.float)

        return x1, x2, x3, x4, x5


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group

from pathlib import Path
import argparse
batch_size = 128
epochs = 256

parser = argparse.ArgumentParser()
parser.add_argument("--mp3_path")
args = parser.parse_args()

mp3_path = Path(args.mp3_path)

files = sorted(list(glob(str(mp3_path / "*.npy"))))

_train, test = train_test_split(files, test_size=0.05, random_state=1337)

train, val = train_test_split(_train, test_size=0.05, random_state=1337)

train_data = AudioDataset(train, augment=True)
test_data = AudioDataset(test, augment=False)
val_data = AudioDataset(val, augment=False)

train_loader = DataLoader(
    train_data, batch_size=batch_size, num_workers=0, shuffle=True
)
val_loader = DataLoader(
    val_data, batch_size=batch_size, num_workers=0, shuffle=True
)
test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=0
)
    

model = Cola()
print(model)
logger = TensorBoardLogger(
    save_dir=".",
    name="lightning_logs",
)

checkpoint_callback = ModelCheckpoint(
    monitor="valid_acc", mode="max", filepath="models/", prefix="encoder"
)

trainer = pl.Trainer(
    max_epochs=epochs,
    gpus=1,
    logger=logger
    # checkpoint_callback=checkpoint_callback,
    # callbacks=[DecayLearningRate()]
)
trainer.fit(model, train_loader, val_loader)

trainer.test(test_dataloaders=test_loader)