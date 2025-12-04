import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from sklearn.metrics import classification_report

from utils.util import init_log, progress_bar

import matplotlib.pyplot as pl
from sklearn import metrics
import numpy as np

import copy
from Models.OTEModel import Model,MAMLModel
from sam import SAM
from bypass_bn import enable_running_stats, disable_running_stats

def smooth_crossentropy(pred, gold, smoothing=0.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def cross_entropy( pred, target_prob ):
    log_prob = F.log_softmax(pred, dim=1)
    return F.kl_div(input=log_prob, target=target_prob, reduction='none').sum(-1)

class AdaptiveAdversrialCrossEntropy(nn.Module):
    def __init__(self, th=0, mode='ada' ): # mode: 'ada', 'rand', 'const'
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.th = th
        self.mode = mode.lower()

    def forward( self, pred, target ):
        with torch.no_grad():
            onehot = F.one_hot( target, num_classes=pred.shape[1] )
            pred_prob = self.softmax( pred.detach() )
            pp = ( pred_prob * onehot ).sum(dim=1, keepdim=True)
            np = pred_prob * (1-onehot)

            dp = F.relu( pp - self.th )
            pp_adv = pp-dp

            if( self.mode == 'ada' ):
                ad = np
            elif( self.mode == 'rand' ):
                ad = (1-torch.rand( pred.shape, device=pred.device)) * (1-onehot)
            elif( self.mode == 'const' ):
                ad = (1-onehot)

            np_adv = ad / ad.sum(dim=1, keepdim=True)

            target_prob = pp_adv * onehot + np_adv

        return cross_entropy( pred, target_prob ) 

def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar() 

    if title is not None:
        pl.title(title)
        
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45)
    pl.yticks(num_local, axis_labels)
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")
    pl.show()

class Trainer():

    def __init__(self, config, model, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.model = model.to(device)
        self.device = device
       
        self.creterion = torch.nn.CrossEntropyLoss()
        self.aa_creterion = AdaptiveAdversrialCrossEntropy(th=0.5, mode='ada')
        # define optimizer
        self.raw_optimizer = torch.optim.SGD(list(self.model.parameters()), lr=config.lr, momentum=config.momentum, weight_decay=config.wd)
        self.scheduler = MultiStepLR(self.raw_optimizer, milestones=[20, 30, 40], gamma=0.1)
        
    def preTrain(self,train_loader,epoch, task_lr, inner_steps):
        meta_loss = 0.0
        correct = 0
        total = 0

        base_optimizer = torch.optim.SGD
        for i, data in enumerate(train_loader):
            # Split the batch into support and query sets
            labels, imgs, texts = data[0].cuda(), data[1].cuda(), data[2].cuda()
            support_size = len(imgs) // 2
            support_imgs, support_texts, support_targets = imgs[:support_size], texts[:support_size], labels[:support_size]
            query_imgs, query_texts, query_targets = imgs[support_size:], texts[support_size:], labels[support_size:]
            # Initialize meta gradients
            self.raw_optimizer.zero_grad()

            # Copy model for inner loop
            fast_model = MAMLModel(self.model.backbone, self.config.num_labels).to(self.device)
            fast_model.load_state_dict(self.model.state_dict())
            fast_model.train()
            task_optimizer = SAM(fast_model.parameters(), base_optimizer, rho=0.2, lr=task_lr, momentum=0.9, weight_decay=0.0005)
            # Inner loop: task-specific adaptation
            for _ in range(inner_steps):
                task_optimizer.zero_grad()

                enable_running_stats(fast_model)
                support_img_logits, support_res_logits, support_text_logits, support_mul_logits = fast_model(support_imgs, support_texts, support_targets)
                support_mul_loss = self.aa_creterion(support_mul_logits, support_targets)
                support_mul_loss.mean().backward()
                task_optimizer.first_step(use_grad_norm=False, zero_grad=True)

                disable_running_stats(fast_model)
                support_img_logits, support_res_logits, support_text_logits, support_mul_logits = fast_model(support_imgs, support_texts, support_targets)
                loss = smooth_crossentropy(support_mul_logits, support_targets)
                loss.mean().backward()
                task_optimizer.second_step(zero_grad=True)

            # Outer loop: meta-optimization
            fast_model.eval()
            with torch.no_grad():
                query_img_logits, query_res_logits, query_text_logits, query_mul_logits = fast_model(query_imgs, query_texts,None)
                query_mul_loss = smooth_crossentropy(query_mul_logits, query_targets)
                meta_loss += query_mul_loss.sum().item()
    
                # Calculate accuracy
                _, predicted = torch.max(query_mul_logits, 1)
                total += query_targets.size(0)
                correct += (predicted == query_targets).sum().item()

            # Update meta-optimizer
            self.raw_optimizer.step()
        
        # 指标 可返回
        # train_losses.append(meta_loss / len(dataloader))
        # train_accuracies.append(100 * correct / total)
        print(f'PreTrain Epoch [{epoch+1}/{self.config.pre_train_epoch}], Meta Loss: {meta_loss/len(train_loader):.4f}, Meta Accuracy: {100 * correct / total:.2f}%')  

    def train(self, train_loader,nowEpoch):
            self.model.train()
            self.scheduler.step()
            
            train_loss_list = []
            train_acc_list = []
                
            train_loss = 0
            train_correct = 0
            train_total = 0

            for i, data in enumerate(train_loader):
                labels, imgs, texts = data[0].cuda(), data[1].cuda(), data[2].cuda()
                batch_size = imgs.size(0)
                self.raw_optimizer.zero_grad()
                
                img_logits, res_logits, text_logits, mul_logits = self.model(imgs, texts,labels)
                
                # calculate loss
                train_img_loss = self.creterion(img_logits, labels)
                train_res_loss = self.creterion(res_logits, labels)
                train_text_loss = self.creterion(text_logits, labels)
                train_mul_loss = self.creterion(mul_logits, labels)
                
                total_loss = train_img_loss + train_res_loss + train_mul_loss + train_text_loss

                train_total += batch_size
                train_loss += train_mul_loss.item() * batch_size
                _, mul_predicts = torch.max(mul_logits, 1)

                # Calculate
                train_correct += torch.sum(mul_predicts.data == labels.data)
                
                # backward
                total_loss.backward()
                self.raw_optimizer.step()
                
                # progress_bar(i, len(train_loader), 'train')
                # print(f'Batch {i}/{len(train_loader)}   Loss: {train_mul_loss.item()}')
            train_loss = train_loss / train_total
            train_acc = float(train_correct / train_total)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                nowEpoch,
                train_loss,
                train_acc,
                train_total))

    def valid(self, val_loader,nowEpoch):
        self.model.eval()

        test_loss_list = []
        test_acc_list = []
        
        test_loss = 0
        test_correct = 0
        test_total = 0

        for i, data in enumerate(val_loader):
            with torch.no_grad():
                labels, imgs, texts = data[0].cuda(), data[1].cuda(), data[2].cuda()
                batch_size = imgs.size(0)

                img_logits, res_logits, text_logits, mul_logits = self.model(imgs, texts,None)
                # calculate loss
                vale_res_loss = self.creterion(mul_logits, labels)

                test_total += batch_size
                test_loss += vale_res_loss.item() * batch_size
                _, mul_predicts = torch.max(mul_logits, 1)
                
                test_correct += torch.sum(mul_predicts.data == labels.data)
                # progress_bar(i, len(val_loader), 'eval val set')
        test_loss = test_loss / test_total
        test_acc = float(test_correct) / test_total
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                nowEpoch,
                test_loss,
                test_acc,
                test_total))
            
        return test_acc
            
    def predict(self, test_loader):
        self.model.eval()
        pred_labels = []
        true_labels = []
        print("********************PredictStart************************")
        for i, data in enumerate(test_loader):
            with torch.no_grad():
                labels, imgs, texts = data[0].cuda(), data[1].cuda(), data[2].cuda()
                batch_size = imgs.size(0)

                img_logits, res_logits, text_logits, mul_logits = self.model(imgs, texts,None)
                _, mul_predicts = torch.max(mul_logits, 1)
                true_labels.extend(labels.tolist())
                pred_labels.extend(mul_predicts.data.tolist())
        print("********************PredictEnd************************")
        print("--------------------Result----------------------------")
        print(classification_report(true_labels, pred_labels, digits=4))
        plot_matrix(true_labels,pred_labels,[0, 1, 2, 3], title='confusion_matrix_svc', axis_labels=['Blight', 'Common_Rust', 'Gray_Leaf_Spot','Healthy'])