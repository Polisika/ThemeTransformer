"""Theme Transformer Training Code

usage: inference.py [-h] [--model_path MODEL_PATH] [--theme THEME]
                    [--seq_length SEQ_LENGTH] [--seed SEED]
                    [--out_midi OUT_MIDI] [--cuda] [--max_len MAX_LEN]
                    [--temp TEMP]
  --model_path MODEL_PATH   model file
  --theme THEME             theme file
  --seq_length SEQ_LENGTH   generated seq length
  --seed SEED               random seed (set to -1 to use random seed) (change different if the model stucks)
  --out_midi OUT_MIDI       output midi file
  --cuda                    use CUDA
  --max_len MAX_LEN         number of tokens to predict
  --temp TEMP               temperature

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

device_str = "cuda:0"

import torch
import torch.optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from mymodel import myLM
from parse_arg import *
from preprocess.music_data import getMusicDataset
from preprocess.vocab import Vocab
from randomness import set_global_random_seed


class ThemeTransformer(pl.LightningModule):
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset = getMusicDataset(
            pkl_path="data_pkl/val_seg2_512.pkl", args=args, vocab=self.vocab
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=4)
        return val_loader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # dataset
        train_dataset = getMusicDataset(
            pkl_path="data_pkl/train_seg2_512.pkl", args=args, vocab=self.vocab
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=32,  # args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return train_loader

    def __init__(self, vocab, d_model=256, num_encoder_layers=6, xorpattern=(0, 0, 0, 1, 1, 1)):
        super().__init__()
        self.transformer = myLM(vocab.n_tokens,
                                d_model=d_model,
                                num_encoder_layers=num_encoder_layers,
                                xorpattern=xorpattern)
        self.total_acc = 0
        self.total_loss = 0
        self.train_step = 0
        self.vocab = vocab

        self.automatic_optimization = False
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def configure_optimizers(self):
        t = torch.optim.Adam(self.transformer.parameters(), lr=args.lr)
        return [t], \
               [torch.optim.lr_scheduler.CosineAnnealingLR(t, T_max=args.max_step, eta_min=args.lr_min)]

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        data["src_msk"] = data["src_msk"].bool()
        data["tgt_msk"] = data["tgt_msk"].bool()

        tgt_input_msk = data["tgt_msk"][:, :-1]
        tgt_output_msk = data["tgt_msk"][:, 1:]

        data["src"] = data["src"].permute(1, 0)
        data["tgt"] = data["tgt"].permute(1, 0)
        data["tgt_theme_msk"] = data["tgt_theme_msk"].permute(1, 0)

        fullsong_input = data["tgt"][:-1, :]
        fullsong_output = data["tgt"][1:, :]

        att_msk = self.transformer.transformer_model.generate_square_subsequent_mask(
            fullsong_input.shape[0]
        ).to(self.device)

        output = self.transformer(
            src=data["src"],
            tgt=fullsong_input,
            tgt_mask=att_msk,
            tgt_label=data["tgt_theme_msk"][:-1, :],
            src_key_padding_mask=data["src_msk"],
            tgt_key_padding_mask=tgt_input_msk,
            memory_mask=None,
        )

        loss = self.criterion(output.view(-1, self.vocab.n_tokens), fullsong_output.reshape(-1))

        predict = output.view(-1, self.vocab.n_tokens).argmax(dim=-1)

        correct = predict.eq(fullsong_output.reshape(-1))
        correct = torch.sum(correct * (~tgt_output_msk).reshape(-1).float()).item()
        correct = correct / torch.sum((~tgt_output_msk).reshape(-1).float()).item()
        self.total_acc += correct

        print("Acc : {:.2f} ".format(correct), end="")

        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), args.clip)
        optimizer.step()

        if self.train_step < args.warmup_step:
            curr_lr = args.lr * self.train_step / args.warmup_step
            optimizer.param_groups[0]["lr"] = curr_lr
        else:
            scheduler.step()

        self.total_loss += loss.item()

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            "Loss : {:.2f} lr:{:.4f} ".format(
                loss.item(), curr_lr
            ),
            end="\r",
        )

        self.train_step += 1
        self.log('train_loss', loss)
        self.log('lr', curr_lr)

    def validation_step(self, data, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        data["src_msk"] = data["src_msk"].bool()
        data["tgt_msk"] = data["tgt_msk"].bool()

        tgt_input_msk = data["tgt_msk"][:, :-1]
        tgt_output_msk = data["tgt_msk"][:, 1:]

        data["src"] = data["src"].permute(1, 0)
        data["tgt"] = data["tgt"].permute(1, 0)
        data["tgt_theme_msk"] = data["tgt_theme_msk"].permute(1, 0)

        fullsong_input = data["tgt"][:-1, :]
        fullsong_output = data["tgt"][1:, :]

        att_msk = self.transformer.transformer_model.generate_square_subsequent_mask(
            fullsong_input.shape[0]
        ).to(self.device)

        output = self.transformer(
            src=data["src"],
            tgt=fullsong_input,
            tgt_mask=att_msk,
            tgt_label=data["tgt_theme_msk"][:-1, :],
            src_key_padding_mask=data["src_msk"],
            tgt_key_padding_mask=tgt_input_msk,
            memory_mask=None,
        )

        loss = self.criterion(
            output.view(-1, self.vocab.n_tokens), fullsong_output.reshape(-1)
        )

        predict = output.view(-1, self.vocab.n_tokens).argmax(dim=-1)
        correct = predict.eq(fullsong_output.reshape(-1))
        correct = torch.sum(correct * (~tgt_output_msk).reshape(-1).float()).item()
        correct = correct / torch.sum((~tgt_output_msk).reshape(-1).float()).item()

        self.total_acc += correct
        self.total_loss += loss.item()
        self.log('val_loss', loss)


# Set the random seed manually for reproducibility.
set_global_random_seed(args.seed)

if __name__ == '__main__':
    # create vocab
    myvocab = Vocab()

    model = ThemeTransformer(myvocab)
    trainer = Trainer(devices=list(range(1, 8)), accelerator='gpu')
    trainer.fit(model)
    trainer.save_checkpoint("model")
