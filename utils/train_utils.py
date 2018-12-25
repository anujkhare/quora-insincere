from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import torch


def prep_inputs(batch, device):
    inputs = Variable(batch['sentence1'].cuda(device=device))
    label = Variable(batch['label']).cuda(device=device)

    return inputs, label


def evaluate(model, dataloader, loss_func, device, n_batches=10):
    loss = 0
    accuracy = 0
    model.train(False)

    for ix, batch in enumerate(dataloader):
        if ix >= n_batches:
            break
        
        inputs, label = prep_inputs(batch, device=device)
        
        predicted = model(inputs)

        loss += loss_func(predicted, label)
        prediction = torch.argmax(predicted, dim=1)

        # TODO: get the confusion here
        accuracy += (torch.sum(prediction == label).data.cpu().numpy() / len(prediction))

    accuracy /= n_batches
    loss /= n_batches
    
    model.train(True)
    
    return loss, accuracy


def train(
    model,
    dataloader_train, dataloader_val,
    loss_func, optimizer,
    device,
    val_every=1000, n_val=100,
    n_epochs=8,
    iter_start=0, epoch_start=0,
    writer=None
) -> None:
    if writer is None:
        print('no tensorboard logging')
        
    epoch = epoch_start
    iteration = iter_start

    while epoch < n_epochs:
        model.train(True)

        for iteration, batch in enumerate(dataloader_train):
            inputs, label = prep_inputs(batch, device=device)

            predicted = model(inputs)

            optimizer.zero_grad()
            loss_train = loss_func(predicted, label)
            loss_train.backward()
            optimizer.step()


            # Log to tensorboard
            iter_total = (epoch * len(dataloader_train)) + iteration
            if writer is not None:
                writer.add_scalar('train.loss', loss_train.data.cpu().numpy(), iter_total)

            # Calculate validation accuracy
            if iteration > 0 and iteration % val_every == 0:
                loss_val, acc_val = evaluate(model, dataloader_val, device=device, loss_func=loss_func, n_batches=n_val)

                s = "Epoch: {}, {:.2f}%: train loss: {}, validation loss: {}, validation acc: {}".format(
                    epoch, (iteration / len(dataloader_train)) * 100, loss_train.data.cpu().numpy(), loss_val, acc_val
                )
                print(s)
                if writer is not None:
                    writer.add_scalar('val.loss', loss_val, iter_total)
                    writer.add_scalar('val.acc', acc_val, iter_total)

        print('\n------------------------------------------------------------------------------------------------------------')
        print("Epoch:", epoch + 1, "label accuracy:", acc_val)
        print('------------------------------------------------------------------------------------------------------------\n')

        torch.save(model.state_dict(), f=os.path.join(model_dir, '{}_{}_{}.pt'.format(model_str, epoch, iteration)))
        epoch += 1