import os
import torch
import time
import torch.nn as nn
import numpy as np
import wandb
import json
import random
import torchsummary
from model import ModelCNN, ModelResNet18
from model import Config
from dataset import get_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device(gpu_id):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda', gpu_id)
    else:
        device = torch.device('cpu')
    return device


def main():
    config = Config()
    print(json.dumps(config.settings, indent=4))
    if config.settings["dataset"] == "CIFAR10":
        model = ModelCNN(config)
    else:
        model = ModelResNet18(config)
    set_seed(config.seed)
    device = get_device(config.gpu_id)
    print("using {}".format(device))
    num_epoches = config.settings["epochs"]
    lr = config.settings["init_lr"]
    train_loader, test_loader = get_dataset(config)
    model.to(device)
    print(model)
    torchsummary.summary(model, (3, 32, 32))

    optimizer = None
    if config.settings["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config.settings["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.settings["lr_scheduler_step_size"], gamma=0.1)
    loss_fun = nn.CrossEntropyLoss().to(device)

    record_t0 = time.time()
    record_time_epoch_step = record_t0

    pt_folder = f"./saves/{config.prob}/"
    if not os.path.exists(pt_folder):
        os.makedirs(pt_folder)
        print(f"Created folder {pt_folder}")
    avg_accuracy_train, avg_accuracy_test = 0, 0
    best_test_accuracy = -1
    best_test_epoch = -1

    for epoch in range(1, num_epoches + 1):
        accuracy_count = []
        loss_list = []
        model.train()
        for batch_id, (x_batch, y_labels) in enumerate(train_loader):
            x_batch, y_labels = x_batch.clone().to(device), y_labels.clone().to(device)
            output_y = model(x_batch)
            # print("x_batch", x_batch.shape, "y_labels", y_labels.shape)
            # print(output_y.shape)
            # print(output_y[:5])
            # print(y_labels[:])
            # print(output_y.shape[-1])
            # assert ((0 <= y_labels) & (y_labels < output_y.shape[-1])).all()
            loss = loss_fun(output_y, y_labels)
            # print(loss)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, y_pred = torch.max(output_y.data, 1)
            correct_match = (y_labels == y_pred)
            accuracy = float(torch.sum(correct_match)) / x_batch.shape[0]
            accuracy_count.append(accuracy)
        avg_accuracy_train = sum(accuracy_count) / len(accuracy_count)
        avg_loss = sum(loss_list) / len(loss_list)



        model.eval()
        with torch.no_grad():
            accuracy_count = []
            for batch_id, (x_batch, y_labels) in enumerate(test_loader):
                x_batch, y_labels = x_batch.clone().to(device), y_labels.clone().to(device)
                output_y = model(x_batch)
                _, y_pred = torch.max(output_y.data, 1)
                correct_match = (y_labels == y_pred)
                accuracy = float(torch.sum(correct_match)) / x_batch.shape[0]
                accuracy_count.append(accuracy)

            avg_accuracy_test = sum(accuracy_count) / len(accuracy_count)
            if avg_accuracy_test > best_test_accuracy:
                best_test_accuracy = avg_accuracy_test
                best_test_epoch = epoch
                pt_save_path = f"{pt_folder}/best.pt"
                checkpoint_info = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_loss,
                }
                torch.save(checkpoint_info, pt_save_path)
        # print(f"avg accuracy test = {avg_accuracy_test:.6f}")

        record_time_epoch_step_tmp = time.time()
        info_epoch = f'Epoch:{epoch:04d}/{num_epoches:04d}  train loss:{avg_loss:.4e}  train_accuracy:{avg_accuracy_train:.4f}  test_accuracy:{avg_accuracy_test:.4f}  '
        info_extended = f'lr:{optimizer.param_groups[0]["lr"]:.9e}  time:{(record_time_epoch_step_tmp - record_time_epoch_step):.2f}s  time total:{((record_time_epoch_step_tmp - record_t0) / 60.0):.2f}min  time remain:{((record_time_epoch_step_tmp - record_t0) / 60.0 / epoch * (num_epoches - epoch)):.2f}min'
        record_time_epoch_step = record_time_epoch_step_tmp
        print(info_epoch + info_extended)
        wandb.log(
            {'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_accuracy_train, 'test_accuracy': avg_accuracy_test, 'lr': optimizer.param_groups[0]["lr"]})
        if epoch % 50 == 0 or epoch == 1:
            pt_save_path = f"{pt_folder}/{epoch:04d}.pt"
            checkpoint_info = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_loss,
            }
            torch.save(checkpoint_info, pt_save_path)
        scheduler.step()

    print(f"Best accuracy on the test set: {best_test_accuracy} (epoch={best_test_epoch})")

    checkpoint = torch.load(f"{pt_folder}/best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    true_positives = np.zeros(config.settings["num_class"])
    false_positives = np.zeros(config.settings["num_class"])
    true_negatives = np.zeros(config.settings["num_class"])
    false_negatives = np.zeros(config.settings["num_class"])

    with torch.no_grad():
        accuracy_count = []
        for batch_id, (x_batch, y_labels) in enumerate(test_loader):
            x_batch, y_labels = x_batch.clone().to(device), y_labels.clone().to(device)
            output_y = model(x_batch)
            _, y_pred = torch.max(output_y.data, 1)
            correct_match = (y_labels == y_pred)
            accuracy = float(torch.sum(correct_match)) / x_batch.shape[0]
            accuracy_count.append(accuracy)

            for i in range(config.settings["num_class"]):
                true_positives[i] += ((y_labels == i) & (y_pred == i)).sum().item()
                false_positives[i] += ((y_labels != i) & (y_pred == i)).sum().item()
                true_negatives[i] += ((y_labels != i) & (y_pred != i)).sum().item()
                false_negatives[i] += ((y_labels == i) & (y_pred != i)).sum().item()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1_score = f1_score.mean()

    print("Precision Distribution:", precision)
    print("Recall Distribution:", recall)
    print("F1 Score Distribution:", f1_score)
    print(f"Precision: {avg_precision:.6f}")
    print(f"Recall: {avg_recall:.6f}")
    print(f"F1 Score: {avg_f1_score:.6f}")


if __name__ == "__main__":
    # with wandb.init(project='CSCI646-CNN', name='CIFAR10-CNN'):
    #     main()
    with wandb.init(project='CSCI646-CNN', name='CIFAR100-ResNet18'):
        main()
