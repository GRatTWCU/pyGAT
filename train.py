from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20, help='Patience')
parser.add_argument('--trajectory_prediction', action='store_true', default=False, 
                    help='Enable trajectory prediction mode with ADE/FDE evaluation.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def compute_ade(predicted, ground_truth):
    """
    Compute Average Displacement Error (ADE)
    Args:
        predicted: [batch_size, seq_len, 2] predicted trajectories
        ground_truth: [batch_size, seq_len, 2] ground truth trajectories
    Returns:
        ADE value
    """
    displacement = torch.sqrt(torch.sum((predicted - ground_truth) ** 2, dim=-1))
    return torch.mean(displacement)


def compute_fde(predicted, ground_truth):
    """
    Compute Final Displacement Error (FDE)
    Args:
        predicted: [batch_size, seq_len, 2] predicted trajectories
        ground_truth: [batch_size, seq_len, 2] ground truth trajectories
    Returns:
        FDE value
    """
    final_displacement = torch.sqrt(torch.sum((predicted[:, -1, :] - ground_truth[:, -1, :]) ** 2, dim=-1))
    return torch.mean(final_displacement)


def evaluate_trajectory_metrics(output, labels):
    """
    Evaluate trajectory prediction using ADE and FDE
    Assumes output and labels are trajectory data with shape [N, seq_len, 2]
    """
    if len(output.shape) == 2:  # If output is flattened, reshape it
        # Assuming output is [N, seq_len*2], reshape to [N, seq_len, 2]
        seq_len = output.shape[1] // 2
        output = output.view(-1, seq_len, 2)
        labels = labels.view(-1, seq_len, 2)
    
    ade = compute_ade(output, labels)
    fde = compute_fde(output, labels)
    
    return ade, fde


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1 if not args.trajectory_prediction else features.shape[1], 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1 if not args.trajectory_prediction else features.shape[1], 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    
    if args.trajectory_prediction:
        # Use MSE loss for trajectory prediction
        loss_train = F.mse_loss(output[idx_train], labels[idx_train])
        ade_train, fde_train = evaluate_trajectory_metrics(output[idx_train], labels[idx_train])
    else:
        # Use NLL loss for classification
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
    
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    if args.trajectory_prediction:
        loss_val = F.mse_loss(output[idx_val], labels[idx_val])
        ade_val, fde_val = evaluate_trajectory_metrics(output[idx_val], labels[idx_val])
        
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'ade_train: {:.4f}'.format(ade_train.data.item()),
              'fde_train: {:.4f}'.format(fde_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'ade_val: {:.4f}'.format(ade_val.data.item()),
              'fde_val: {:.4f}'.format(fde_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    
    if args.trajectory_prediction:
        loss_test = F.mse_loss(output[idx_test], labels[idx_test])
        ade_test, fde_test = evaluate_trajectory_metrics(output[idx_test], labels[idx_test])
        
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "ADE= {:.4f}".format(ade_test.data.item()),
              "FDE= {:.4f}".format(fde_test.data.item()))
        
        return {
            'loss': loss_test.data.item(),
            'ADE': ade_test.data.item(),
            'FDE': fde_test.data.item()
        }
    else:
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))
        
        return {
            'loss': loss_test.data.item(),
            'accuracy': acc_test.data.item()
        }


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
test_results = compute_test()

# Save results to file
if args.trajectory_prediction:
    with open('trajectory_results.txt', 'w') as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Test Loss: {test_results['loss']:.4f}\n")
        f.write(f"Test ADE: {test_results['ADE']:.4f}\n")
        f.write(f"Test FDE: {test_results['FDE']:.4f}\n")
        f.write(f"Total Training Time: {time.time() - t_total:.4f}s\n")
else:
    with open('classification_results.txt', 'w') as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Test Loss: {test_results['loss']:.4f}\n")
        f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Total Training Time: {time.time() - t_total:.4f}s\n")

print("Results saved to file!")