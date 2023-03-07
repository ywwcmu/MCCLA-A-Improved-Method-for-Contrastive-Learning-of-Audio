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
        x1, x2, x3, x4= x

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

        x11 = self.linear(x1)
        x22 = self.linear(x2)
        x33 = self.linear(x3)
        x44 = self.linear(x4)

        return x1, x2, x3, x4, x11, x22, x33, x44


    def training_step(self, x, batch_idx):
        x1, x2, x3, x4, x11, x22, x33, x44 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat12 = torch.mm(x11, x2.t())
        y_hat13 = torch.mm(x11, x3.t())
        y_hat14 = torch.mm(x11, x4.t())
        y_hat23 = torch.mm(x22, x3.t())
        y_hat24 = torch.mm(x22, x4.t())
        y_hat34 = torch.mm(x33, x4.t())

        diag12 = torch.diag(y_hat12)
        diag13 = torch.diag(y_hat13)
        diag14 = torch.diag(y_hat14)
        diag23 = torch.diag(y_hat23)
        diag24 = torch.diag(y_hat24)
        diag34 = torch.diag(y_hat34)
        
        diag = torch.minimum(torch.minimum(torch.minimum(diag12,diag13),torch.minimum(diag14,diag23)),torch.minimum(diag24,diag34))

        y_hat = torch.maximum(torch.maximum(torch.maximum(y_hat12,y_hat13),torch.maximum(y_hat14,y_hat23)),torch.maximum(y_hat24,y_hat34))
        y_hat_new = torch.sub(y_hat,torch.diag_embed(torch.diag(y_hat)))
        y_hat_new = torch.add(y_hat_new,torch.diag_embed(diag))

        loss = F.cross_entropy(y_hat_new, y)

        _, predicted12 = torch.max(y_hat12, 1)
        _, predicted13 = torch.max(y_hat13, 1)
        _, predicted14 = torch.max(y_hat14, 1)
        _, predicted23 = torch.max(y_hat23, 1)
        _, predicted24 = torch.max(y_hat24, 1)
        _, predicted34 = torch.max(y_hat34, 1)
        

        acc1 = (predicted12 == y).double().mean()
        acc2 = (predicted13 == y).double().mean()
        acc3 = (predicted23 == y).double().mean()
        acc4 = (predicted14 == y).double().mean()
        acc5 = (predicted24 == y).double().mean()
        acc6 = (predicted34 == y).double().mean()

        acc = (acc1+acc2+acc3+acc4+acc5+acc6)/6

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, x, batch_idx):
        x1, x2, x3, x4, x11, x22, x33, x44 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat12 = torch.mm(x11, x2.t())
        y_hat13 = torch.mm(x11, x3.t())
        y_hat14 = torch.mm(x11, x4.t())
        y_hat23 = torch.mm(x22, x3.t())
        y_hat24 = torch.mm(x22, x4.t())
        y_hat34 = torch.mm(x33, x4.t())

        diag12 = torch.diag(y_hat12)
        diag13 = torch.diag(y_hat13)
        diag14 = torch.diag(y_hat14)
        diag23 = torch.diag(y_hat23)
        diag24 = torch.diag(y_hat24)
        diag34 = torch.diag(y_hat34)
        
        diag = torch.minimum(torch.minimum(torch.minimum(diag12,diag13),torch.minimum(diag14,diag23)),torch.minimum(diag24,diag34))

        y_hat = torch.maximum(torch.maximum(torch.maximum(y_hat12,y_hat13),torch.maximum(y_hat14,y_hat23)),torch.maximum(y_hat24,y_hat34))
        y_hat_new = torch.sub(y_hat,torch.diag_embed(torch.diag(y_hat)))
        y_hat_new = torch.add(y_hat_new,torch.diag_embed(diag))

        loss = F.cross_entropy(y_hat_new, y)

        _, predicted12 = torch.max(y_hat12, 1)
        _, predicted13 = torch.max(y_hat13, 1)
        _, predicted14 = torch.max(y_hat14, 1)
        _, predicted23 = torch.max(y_hat23, 1)
        _, predicted24 = torch.max(y_hat24, 1)
        _, predicted34 = torch.max(y_hat34, 1)
        

        acc1 = (predicted12 == y).double().mean()
        acc2 = (predicted13 == y).double().mean()
        acc3 = (predicted23 == y).double().mean()
        acc4 = (predicted14 == y).double().mean()
        acc5 = (predicted24 == y).double().mean()
        acc6 = (predicted34 == y).double().mean()

        acc = (acc1+acc2+acc3+acc4+acc5+acc6)/6

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def test_step(self, x, batch_idx):
        x1, x2, x3, x4, x11, x22, x33, x44 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat12 = torch.mm(x11, x2.t())
        y_hat13 = torch.mm(x11, x3.t())
        y_hat14 = torch.mm(x11, x4.t())
        y_hat23 = torch.mm(x22, x3.t())
        y_hat24 = torch.mm(x22, x4.t())
        y_hat34 = torch.mm(x33, x4.t())

        diag12 = torch.diag(y_hat12)
        diag13 = torch.diag(y_hat13)
        diag14 = torch.diag(y_hat14)
        diag23 = torch.diag(y_hat23)
        diag24 = torch.diag(y_hat24)
        diag34 = torch.diag(y_hat34)
        
        diag = torch.minimum(torch.minimum(torch.minimum(diag12,diag13),torch.minimum(diag14,diag23)),torch.minimum(diag24,diag34))

        y_hat = torch.maximum(torch.maximum(torch.maximum(y_hat12,y_hat13),torch.maximum(y_hat14,y_hat23)),torch.maximum(y_hat24,y_hat34))
        y_hat_new = torch.sub(y_hat,torch.diag_embed(torch.diag(y_hat)))
        y_hat_new = torch.add(y_hat_new,torch.diag_embed(diag))

        loss = F.cross_entropy(y_hat_new, y)

        _, predicted12 = torch.max(y_hat12, 1)
        _, predicted13 = torch.max(y_hat13, 1)
        _, predicted14 = torch.max(y_hat14, 1)
        _, predicted23 = torch.max(y_hat23, 1)
        _, predicted24 = torch.max(y_hat24, 1)
        _, predicted34 = torch.max(y_hat34, 1)
        

        acc1 = (predicted12 == y).double().mean()
        acc2 = (predicted13 == y).double().mean()
        acc3 = (predicted23 == y).double().mean()
        acc4 = (predicted14 == y).double().mean()
        acc5 = (predicted24 == y).double().mean()
        acc6 = (predicted34 == y).double().mean()

        acc = (acc1+acc2+acc3+acc4+acc5+acc6)/6

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
        # print(x1.shape)
        # print(x2.shape)

        if self.augment:
            x1 = random_multiply(x1)
            x2 = random_multiply(x2)
            x3 = random_multiply(x3)
            x4 = random_multiply(x4)

        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        x3 = torch.tensor(x3, dtype=torch.float)
        x4 = torch.tensor(x4, dtype=torch.float)

        return x1, x2, x3, x4


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