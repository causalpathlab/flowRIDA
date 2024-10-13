import logging

import pandas as pd
import torch

from . import datautil as dutil
from . import model as md

from tqdm import tqdm


class gcan(object):
    def __init__(self, data, wdir):
        self.data = data
        self.wdir = wdir
        logging.basicConfig(
            filename=self.wdir + "results/gcan_train.log",
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def set_nn_params(self, params: dict):
        self.nn_params = params

    def train(self):

        logging.info("Starting training...")

        logging.info(self.nn_params)

        gcan_model = md.ETM(
            self.nn_params["input_dim"],
            self.nn_params["latent_dim"],
            self.nn_params["layers"],
        ).to(self.nn_params["device"])

        logging.info(gcan_model)

        data = dutil.nn_load_data(
            self.data, self.nn_params["device"], self.nn_params["batch_size"]
        )
        

        loss = md.train(
            gcan_model, data, self.nn_params["epochs"], self.nn_params["learning_rate"]
        )

        torch.save(gcan_model.state_dict(), self.wdir + "results/nn_gcan.model")
        df_loss = pd.DataFrame(loss, columns=["loss", "loglik", "kl", "klb"])
        df_loss.to_csv(
            self.wdir + "results/4_gcan_train_loss.txt.gz",
            index=False,
            compression="gzip",
            header=True,
        )
        logging.info("Completed training...model saved in results/nn_gcan.model")

    def eval(self, device, eval_batch_size):

        logging.info("Starting eval...")

        logging.info(self.nn_params)

        gcan_model = md.ETM(
            self.nn_params["input_dim"],
            self.nn_params["latent_dim"],
            self.nn_params["layers"],
        ).to(self.nn_params["device"])

        gcan_model.load_state_dict(
            torch.load(
                self.wdir + "results/nn_gcan.model", map_location=torch.device(device)
            )
        )

        gcan_model.eval()

        logging.info(gcan_model)

        data = dutil.nn_load_data(self.data, device, eval_batch_size)

        df_theta = pd.DataFrame()

        for xx, y in data:
            theta, beta, ylabel = md.predict_batch(gcan_model, xx, y)
            df_theta = pd.concat(
                [df_theta, pd.DataFrame(theta.cpu().detach().numpy(), index=ylabel)],
                axis=0,
            )
        df_theta.to_csv(
            self.wdir + "results/df_theta.txt.gz",
            index=True,
            compression="gzip",
            header=True,
        )

    def plot_loss(self):
        import matplotlib.pylab as plt

        plt.rcParams.update({"font.size": 20})

        loss_f = self.wdir + "results/4_gcan_train_loss.txt.gz"
        fpath = self.wdir + "results/4_gcan_train_loss.png"
        pt_size = 4.0

        data = pd.read_csv(loss_f)
        num_subplots = len(data.columns)
        fig, axes = plt.subplots(
            num_subplots, 1, figsize=(10, 6 * num_subplots), sharex=True
        )

        for i, column in enumerate(data.columns):
            data[[column]].plot(ax=axes[i], legend=None, linewidth=pt_size, marker="o")
            axes[i].set_ylabel(column)
            axes[i].set_xlabel("epoch")
            axes[i].grid(False)

        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()


def create_gcan_object(adata, wdir):
    return gcan(adata, wdir)
