
import argparse
import lightning as L
from Model_Direct_LR import PedalNet
from lightning.pytorch.callbacks import ModelCheckpoint
import os


def main(args):
    model = PedalNet(vars(args))

    latest_checkpoint = ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="epoch",
        save_last=False,
    )



    best = ModelCheckpoint(save_top_k=3, monitor="validation_epoch_mean", auto_insert_metric_name=True,
                           filename='{epoch}-{validation_epoch_mean}', save_last=True)


    new_dir = '' #Save Directory


    trainer = L.Trainer(
        max_epochs=args.max_epochs, accelerator=args.gpus, devices=args.dev, log_every_n_steps=2, callbacks=[best, latest_checkpoint], default_root_dir=new_dir,
        gradient_clip_val=1
    )
    #  set default_root_dir  to change save directory
    if args.checkp != None:
        model=PedalNet.load_from_checkpoint(args.checkp)

    trainer.fit(model, ckpt_path=args.checkp)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--dilation_depth", type=int, default=8)
    parser.add_argument("--num_repeat", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--W_len", type=int, default=4800)
    parser.add_argument("--N_step", type=int, default=64*6)

    parser.add_argument("--max_epochs", type=int, default = 40000)
    parser.add_argument("--gpus", default= "gpu")
    parser.add_argument("--dev", default=[1])

    parser.add_argument("--path_Dataset", default="")  #Dataset Directory
    parser.add_argument("--N_workers", default=8)
    parser.add_argument("--Data_test", default=["p228_1", "pia1"])
    parser.add_argument("--Data_val", default=["p230_1", "cla1"])
    parser.add_argument("--checkp", default=None)
    args = parser.parse_args()
    main(args)