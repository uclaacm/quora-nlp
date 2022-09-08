from constants import *
import torch
from torch import nn
from tqdm import tqdm, tqdm_notebook  # log writer
import sys
from torch.utils.tensorboard import SummaryWriter

# setting path
sys.path.append('../')


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, train_loader, optimizer, loss_criterion, device):
    model.train()  # switch model to train mode
    train_losses = []
    tb_writer = SummaryWriter()

    for epoch in range(EPOCHS):

        progress_bar = tqdm_notebook(train_loader, leave=False)

        losses = []
        accs = []
        total = 0

        for inputs, target in progress_bar: #for BERT there's also an attention mask

            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()
            model.zero_grad()

            output = model(inputs)

            loss = loss_criterion(output.squeeze(), target)
            acc = binary_accuracy(output.squeeze(), target)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 3)  # gradient clipping
            optimizer.step()

            progress_bar.set_description(f'Loss: {loss.item():.3f}')
            # TODO: add validation to the training loop

            loss_val = loss.item()

            losses.append(loss.item())
            accs.append(acc)
            total += 1

            if total % N_EVAL == 0:
                tb_writer.add_scalar("Accuracy (Train)", acc, total)
                tb_writer.add_scalar("Loss (Train)", loss.item(), total)

        epoch_loss = sum(losses) / total
        accuracy = sum(accs) / total
        train_losses.append(epoch_loss)

        tqdm.write(
            f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}\tAccuracy: {accuracy:.3f}')

    tb_writer.flush()
    tb_writer.close()
