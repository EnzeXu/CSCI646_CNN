import torch
import torch.nn as nn

from model import Model
from model import Config
from dataset import get_dataset


def main():
    use_cuda = torch.cuda.is_available()  ## if have gpu or cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    if use_cuda:
        torch.cuda.manual_seed(72)

    ## initialize hyper-parameters
    config = Config()
    model = Model(config)
    num_epoches = config.epoch
    # decay = config.decay
    learning_rate = config.lr

    train_loader, test_loader = get_dataset(config.batch_size)

    model.to(device)

    ## --------------------------------------------------
    ## Step 3: write the LOSS FUNCTION ##
    ## --------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  ## optimizer
    loss_fun = nn.MSELoss() ## cross entropy loss

    ##--------------------------------------------
    ## load checkpoint below if you need
    ##--------------------------------------------
    # if args.load_checkpoint == True:
    ## write load checkpoint code below

    ##  model training
    model = model.train()
    avg_accuracy_train, avg_accuracy_test = 0, 0
    for epoch in range(1, num_epoches + 1):  # 10-50
        accuracy_count = []
        for batch_id, (x_batch, y_labels) in enumerate(train_loader):
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)

            # print(x_batch.shape, y_labels.shape)

            ## feed input data x into model
            output_y = model(x_batch)

            ##---------------------------------------------------
            ## Step 4: write loss function below, refer to tutorial slides
            ##----------------------------------------------------
            loss = loss_fun(output_y, nn.functional.one_hot(y_labels, 10).float())

            ##----------------------------------------
            ## Step 5: write back propagation below
            ##----------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ##------------------------------------------------------
            ## Step 6: get the predict result and then compute accuracy below
            ## please refer to defined _compute_accuracy() above
            ##------------------------------------------------------
            _, y_pred = torch.max(output_y.data, 1)
            # score, idx = torch.max(output.data, 1)
            # print(y_labels.shape, y_pred.shape)
            correct_match = (y_labels == y_pred)
            accuracy = float(torch.sum(correct_match)) / x_batch.shape[0]
            accuracy_count.append(accuracy)
        avg_accuracy_train = sum(accuracy_count) / len(accuracy_count)
        print(f"Epoch {epoch:04d} / {num_epoches:04d}: avg accuracy train = {avg_accuracy_train:.6f}")
        ##----------------------------------------------------------
        ## Step 7: use wandb to visualize the loss blow
        ## if use loss.item(), you may use log txt files to save loss
        ##----------------------------------------------------------

    ## -------------------------------------------------------------------
    ## Step 8: save checkpoint below (optional), every "epoch" save one checkpoint
    ## -------------------------------------------------------------------

    ##----------------------------------------
    ##    Step 9: model testing code below
    ##----------------------------------------
    model.eval()
    with torch.no_grad():
        accuracy_count = []
        for batch_id, (x_batch, y_labels) in enumerate(test_loader):
            x_batch, y_labels = torch.tensor(x_batch).to(device), torch.tensor(y_labels).to(device)
            output_y = model(x_batch)
            _, y_pred = torch.max(output_y.data, 1)
            correct_match = (y_labels == y_pred)
            accuracy = float(torch.sum(correct_match)) / x_batch.shape[0]
            accuracy_count.append(accuracy)

        avg_accuracy_test = sum(accuracy_count) / len(accuracy_count)
    print(f"avg accuracy test = {avg_accuracy_test:.6f}")

if __name__ == "__main__":
    main()