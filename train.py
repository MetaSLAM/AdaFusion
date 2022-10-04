import argparse
import time
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import models
from eval.evaluate import recall_atN
from loader_loss.pairwise_margin import (
    PairwiseData,
    PairwiseMarginLoss,
    hard_mining,
    train_loader_generator,
)
from loader_loss.testdata import get_test_set
from utils import (
    get_logger,
    load_config,
    count_parameters,
    get_current_lr,
    load_checkpoint,
    save_checkpoint,
)


# parameter parser
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--work-path", required=True, type=str, help="working directory")
parser.add_argument("--lr", type=float, help="assign an new learning rate.")
parser.add_argument("--best_prec", type=float, help="assign an new best_prec.")
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
args = parser.parse_args()

# logger, to record messages
logger = get_logger(
    log_file_name=args.work_path + "/log.txt", log_level="DEBUG", logger_name="CIFAR"
)

# tensorBoard writter
writer = SummaryWriter(log_dir=args.work_path + "/event")

# load config from yaml file
config = load_config(args.work_path + "/config.yaml")
logger.info(config)

# variables for recording
best_prec = 0  #


#############################################################
#       The following variables are used GLOBALLY
# tools: ``args``, ``logger``, ``writer``, ``config``
# records: ``best_prec``
#############################################################


def test(test_loader, net, optimizer, epoch, device):
    """Test/Evaluate the network performance (Recall @1). Also save the model."""
    global best_prec
    net.eval()  # change to 'evaluate' stage

    features = []
    logger.info(" === Validation ===")

    # No need for gradient in evaluation stage
    # Calculate features for all items in the test set
    with torch.no_grad():
        for image, pc in test_loader:
            image, pc = image.to(device), pc.to(device)
            batch_feature = net(image, pc)
            batch_feature = batch_feature.detach().cpu().numpy()
            features.append(batch_feature)
    features = np.vstack(features)

    # get some variables
    pos_items = test_loader.dataset.get_pos_items()
    num_of_each_run = test_loader.dataset.get_num_of_each_run()  # [100, 102, ...]
    sum_num_of_each_run = [
        sum(num_of_each_run[:i]) for i in range(len(num_of_each_run))
    ]  # [0, 100, 202, ...]
    run_num = len(num_of_each_run)

    # compute evaluation metric
    recall_1s = []
    meanAPs = []
    pairs = ((i, j) for i in range(run_num) for j in range(i + 1, run_num))
    for i, j in pairs:
        st1 = sum_num_of_each_run[i]
        st2 = sum_num_of_each_run[j]
        end1 = st1 + num_of_each_run[i]
        end2 = st2 + num_of_each_run[j]

        feature_of_two_run = np.vstack((features[st1:end1], features[st2:end2]))
        pos_items_of_two_run = pos_items[(i, j)]

        recall_1 = recall_atN(
            feature_of_two_run, pos_items_of_two_run, N=1, Lp=config.loss.Lp
        )
        recall_1s.append(recall_1)

    # show and record test results
    recall_1 = np.mean(recall_1s)
    meanAP = np.mean(meanAPs)
    logger.info(f"   == test recall@1: {recall_1:.4f}")
    writer.add_scalar("test_recall_1", recall_1)

    # judge best testing
    is_best = recall_1 > best_prec
    if is_best:
        best_prec = recall_1

    # Save checkpoint.
    state = {
        "state_dict": net.state_dict(),
        "best_prec": best_prec,
        "last_epoch": epoch,
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(state, is_best, args.work_path + "/" + config.ckpt_name)
    logger.info(
        f"   == save checkpoint, recall@1={recall_1:.4}, is_best={is_best}, "
        f"best={best_prec:.4} =="
    )

    net.train()  # change to 'train' stage
    return recall_1


def train(loaders, net, criterion, optimizer, lr_scheduler, epoch, device):
    """Train the network for one epoch."""
    train_loader, test_loader = loaders
    start_time = time.time()
    net.train()  # change to 'train' stage

    train_loss_sum = 0
    batch_total_num = len(train_loader)
    logger.info(f" === start Epoch: [{epoch + 1}/{config.epochs}] ===")

    # use batch to iterate through the dataset (one epoch)
    for batch_index, pairs in enumerate(train_loader):
        # ========== Hard mining ==========
        if config.hardM.enabled:
            if batch_index == 0:
                hard_mining_loader = train_loader_generator(train_loader)
            if pairs["pos_pair"][0][0].shape[0] != config.train_batch_size:
                continue  # Skip the last batch
            if batch_index % config.hardM.hardM_freq == 0:
                hard_pair, hard_loss = hard_mining(
                    hard_mining_loader, net, criterion, device, config
                )
                logger.info(
                    f"   == sampling hard sample, top hard loss={hard_loss:.3f} =="
                )
            pairs["hard_pair"] = hard_pair

        # ========== Forward ==========
        loss = 0.0
        for pair_key, pair_data in pairs.items():
            y = 1 if pair_key == "pos_pair" else -1
            x = []  # store x1 and x2, features representation of pairs

            # each pair contains two X=(image, pc)
            for image, pc in pair_data:
                image, pc = image.to(device), pc.to(device)
                x.append(net(image, pc))
            loss += criterion(*x, y)  # of size [N]
        loss = torch.mean(loss)

        # ========== Backward & Update ==========
        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()  # backward
        optimizer.step()  # update weight

        # ========== Counting and Statistics ==========
        train_loss_sum += loss.item()

        # ========== Show Infomation ==========
        if batch_index % config.show_freq == 0:
            logger.info(
                f"   == for step: [{batch_index+1:5}/{batch_total_num:5}], "
                f"train loss: {loss.item():.3f} | "
                f"lr: {get_current_lr(optimizer):.5f}"
            )

        # ========== Eval & Save Checkpoint ==========
        if (batch_index + 1) % config.eval_freq == 0:
            recall_1 = test(test_loader, net, optimizer, epoch, device)
            lr_scheduler.step(recall_1)  # adjust learning rate if no improvement
            writer.add_scalar("learning_rate", get_current_lr(optimizer))

        # <-- end for batch

    # record time for one epoch, train loss and train accuracy
    train_loss_avg = train_loss_sum / batch_total_num
    logger.info(f"   == cost time: {time.time() - start_time:.4f}s")
    logger.info(f"   == average train loss: {train_loss_avg:.3f}")
    writer.add_scalar("train_loss", train_loss_avg, global_step=epoch)

    return train_loss_avg


def main():
    global best_prec
    logger.info("\n\n\n" + "=" * 15 + " New Run " + "=" * 15)

    # define netowrk
    net = models.get_model(config)
    logger.info(net)
    logger.info(f" == total parameters: {count_parameters(net)} ==")

    # CPU or GPU
    device = "cuda" if config.use_gpu else "cpu"
    logger.info(f" == will be trained on device: {device} ==")
    if device == "cuda":  # data parallel for multiple-GPU
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    net.to(device)

    # define loss and optimizer
    criterion = PairwiseMarginLoss(config.loss.a, config.loss.m, config.loss.Lp)
    optimizer = torch.optim.Adam(
        net.parameters(),
        config.optimize.base_lr,
        betas=config.optimize.betas,
        weight_decay=config.optimize.weight_decay,
        amsgrad=config.optimize.amsgrad,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_scheduler.factor,
        patience=config.lr_scheduler.patience,
        cooldown=config.lr_scheduler.cooldown,
        min_lr=config.optimize.base_lr / 10,
    )

    # resume from a checkpoint
    last_epoch = -1
    ckpt_file_name = args.work_path + "/" + config.ckpt_name
    if args.resume:
        best_prec, last_epoch = load_checkpoint(ckpt_file_name, net, optimizer)
        lr_scheduler.step(best_prec)
    # overwrite learning rate
    if args.lr is not None:
        logger.info(f"learning rate is overwritten to {args.lr:.5}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
    # overwrite best prediction
    if args.best_prec is not None:
        logger.info(f"best_prec is overwritten to {args.best_prec:.4}")
        best_prec = args.best_prec

    # load training and testing data loader
    train_loader = DataLoader(
        PairwiseData(config=config),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        get_test_set(config),
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.workers,
    )

    # start training --->
    logger.info("==============  Start Training  ==============\n")
    for epoch in range(last_epoch + 1, config.epochs):
        random_int = torch.randint(200, 7000, (1,)).item()
        logger.info(f" === random number for dataloader this epoch: {random_int} ===")
        train_loader.dataset.shuffle_data(random_int)

        train(
            (train_loader, test_loader),
            net,
            criterion,
            optimizer,
            lr_scheduler,
            epoch,
            device,
        )

    # ---> training finished
    logger.info(f"======== Training Finished.  best_test_acc: {best_prec:.3%} ========")
    writer.close()


if __name__ == "__main__":
    main()
