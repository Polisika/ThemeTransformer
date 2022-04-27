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
import time

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ThemeTransformer.model_definition import ThemeTransformer

from preprocess.vocab import Vocab
from randomness import set_global_random_seed
from parse_arg import get_args
# Set the random seed manually for reproducibility.

if __name__ == '__main__':
    args = get_args()

    set_global_random_seed(args.seed)
    # create vocab
    myvocab = Vocab()

    model = ThemeTransformer(myvocab, args)
    epochs = 15000
    logger = TensorBoardLogger("tensor_board_logs", name=f"model_epochs={epochs}")
    trainer = Trainer(devices=[2, 3],
                      accelerator='gpu',
                      strategy="ddp",
                      max_epochs=epochs,
                      enable_checkpointing=True,
                      check_val_every_n_epoch=10,
                      log_every_n_steps=10,
                      logger=logger)
    start = time.time()
    trainer.fit(model)
    trainer.save_checkpoint(f"model_{epochs}_epochs.ckpt")
    print(f"Training takes {(time.time() - start) / 60 / 60} hours")
