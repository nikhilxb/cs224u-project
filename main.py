import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import utils
from dataset_wrapper import NYT10Dataset
from model_v1 import RelationClassifier

def load_saved_model(filepath, model, optimizer=None):
    state = torch.load(filepath)
    model.load_state_dict(state['state_dict'])
    # Only need to load optimizer if you are going to resume training on the model
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_cuda, best_model_filepath, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            true_y = []
            predictions = []
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], ncols=80):
                if use_cuda:
                #     inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                true_y.extend(labels.tolist())
                predictions.extend(preds.tolist())
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(classification_report(true_y, predictions))

            # deep copy the model
            # TODO: use a better metric than accuracy?
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, best_model_filepath)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, testset_loader, test_size, use_gpu):
    model.train(False)  # Set model to evaluate mode

    predictions = []
    # Iterate over data
    for inputs, labels in tqdm(testset_loader, ncols=60):
        # TODO: wrap them in Variable?
        if use_gpu:
            # inputs = inputs.cuda()
            labels = labels.cuda()

        # forward
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        _, preds = torch.max(outputs.data, 1)
        predictions.extend(preds.tolist())
    return predictions

def init_weights(m):
    # He initialization
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.kaiming_normal_(m.weight)


def run():
    # train_dataset = NYT10Dataset('data/small_train.txt', 'data/relation2id.txt')
    # val_dataset = NYT10Dataset('data/small_val.txt', 'data/relation2id.txt')
    train_dataset = NYT10Dataset('data/train.txt', 'data/relation2id.txt')
    val_dataset = NYT10Dataset('data/val.txt', 'data/relation2id.txt')
    # test_dataset = NYT10Dataset('data/test.txt', 'data/relation2id.txt')

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    # if use_parallel:
    #     print("[Using all the available GPUs]")
    #     inception = nn.DataParallel(inception, device_ids=[0])

    vocab = utils.glove2dict("data/glove.6B.50d.txt")  # dict[word] -> numpy array(embed_dim,)
    rc_model = RelationClassifier(vocab, 50, train_dataset.num_relations(), device=device).to(device)
    rc_model.apply(init_weights)

    def collate_fn(batch):
        X, y = zip(*batch)
        return X, torch.LongTensor(y)

    trainset_loader = DataLoader(train_dataset,
                                batch_size=50,
                                shuffle=True,
                                num_workers=20,
                                collate_fn=collate_fn)
    valset_loader = DataLoader(val_dataset,
                                batch_size=50,
                                shuffle=False,
                                num_workers=10,
                                collate_fn=collate_fn)

    best_model_filepath = 'models/model_best.weighted.1e-1.pth.tar'
    stats_filepath = 'train_log.txt'

    dataloaders = {'train': trainset_loader, 'val': valset_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    loss_weights = torch.ones(train_dataset.num_relations(), device=device)
    loss_weights[0] = 1e-1
    # loss_weights[48] = 1e-2
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizable_params = [param for param in rc_model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(optimizable_params, lr=0.01)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    num_epochs = 20

    # load_saved_model(best_model_filepath, rc_model, optimizer)
    # best_model = rc_model

    best_model = train_model(rc_model,
                                dataloaders,
                                dataset_sizes,
                                criterion,
                                optimizer,
                                exp_lr_scheduler,
                                use_cuda,
                                best_model_filepath,
                                num_epochs)


    predictions = evaluate_model(best_model, valset_loader, len(val_dataset), use_cuda)
    true_y = [y for _, y in val_dataset]
    report = classification_report(true_y, predictions)
    with open(stats_filepath, 'a') as f:
        f.write(report)
    print(report)

    # predictions = evaluate_model(best_model, testset_loader, len(test_dataset), use_cuda)
    # true_y = [y for img, y in test_dataset]
    # print(classification_report(true_y, predictions))

if __name__ == '__main__':
    run()