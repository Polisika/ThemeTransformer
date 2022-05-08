import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from preprocess.vocab import Vocab
from mymodel import myLM
from preprocess.music_data import getMusicDataset
from parse_arg import DefaultTrainArgs


class ThemeTransformer(pl.LightningModule):
    def __init__(self, batch_size=16, d_model=256, num_encoder_layers=6, xorpattern=(0, 0, 0, 1, 1, 1)):
        super().__init__()
        vocab = Vocab()
        self.transformer = myLM(vocab.n_tokens,
                                d_model=d_model,
                                num_encoder_layers=num_encoder_layers,
                                xorpattern=xorpattern)
        self.total_acc = 0
        self.total_loss = 0
        self.train_step = 0
        self.vocab = vocab
        self.args = DefaultTrainArgs()

        self.automatic_optimization = False
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.batch_size = batch_size

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset = getMusicDataset(
            pkl_path="data_pkl/val_seg2_512.pkl", args=self.args, vocab=self.vocab
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return val_loader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # dataset
        train_dataset = getMusicDataset(
            pkl_path="data_pkl/train_seg2_512.pkl", args=self.args, vocab=self.vocab
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,  # args.batch_size, -- 8 default * 7 gpus
            shuffle=True,
            num_workers=4,
        )
        return train_loader

    def configure_optimizers(self):
        t = torch.optim.Adam(self.transformer.parameters(), lr=self.args.lr)
        return [t], \
               [torch.optim.lr_scheduler.CosineAnnealingLR(t, T_max=self.args.max_step, eta_min=self.args.lr_min)]

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

        reshape_mask = (~tgt_output_msk).reshape(-1).float()
        correct = torch.sum(correct * reshape_mask).item()
        correct = correct / torch.sum(reshape_mask).item()
        self.total_acc += correct

#         print("Acc : {:.2f} ".format(correct), end="")

        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), self.args.clip)
        optimizer.step()

        if self.train_step < self.args.warmup_step:
            curr_lr = self.args.lr * self.train_step / self.args.warmup_step
            optimizer.param_groups[0]["lr"] = curr_lr
        else:
            scheduler.step()

        self.total_loss += loss.item()

#         curr_lr = optimizer.param_groups[0]["lr"]
#         print(
#             "Loss : {:.2f} lr:{:.4f} ".format(
#                 loss.item(), curr_lr
#             ),
#             end="\r",
#         )

        self.train_step += 1
        self.log('train_loss', loss, sync_dist=True)
        self.log('lr', curr_lr, sync_dist=True)

        return {
            "loss": loss,
            "log": {"train_loss": loss, "total_acc": self.total_acc},
            "lr": curr_lr
        }

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
        self.log('val_loss', loss, sync_dist=True)
        return {
            "loss": loss.item(),
            "log": {"train_loss": loss.item(), "total_acc": self.total_acc}
        }
