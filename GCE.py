import numpy as np
import scipy.io as sio
import torch
import os
from GraphVAE import VAE_LL_loss
from utils import calculate_MI
from causaleffect import joint_uncond
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix

criterion = nn.CrossEntropyLoss()
class GenerativeCausalExplainer:

    """
    :param classifier: classifier to explain
    :param decoder: decoder model
    :param encoder: encoder model
    :param device: pytorch device object
    """
    def __init__(self, classifier, decoder, encoder,casual_decoder, device):

        # initialize
        super(GenerativeCausalExplainer, self).__init__()
        self.classifier = classifier
        self.decoder = decoder
        self.casual_decoder = casual_decoder
        self.encoder = encoder
        self.device =device


    """
    :param X: training data (not necessarily same as classifier training data)
    :param steps: number of training steps
    :param Nalpha: number of causal latent factors
    :param lam: regularization parameter
    :param batch_size: batch size for training
    :param lr: learning rate for adam optimizer
    """

    def train(self, data, eval_data, test_data,
              steps = 200,
              Nalpha = 20,
              lam = 0.05,
              batch_size = 32,
              lr = 0.0001,
              ce = 0.001,
              r = 0.001):
        print(Nalpha)
        # initialize for training
        opt_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        self.opt = torch.optim.Adam(opt_params, lr=lr , weight_decay=5e-4)
        params = list(self.classifier.parameters()) + list(self.casual_decoder.parameters())
        optimizer = torch.optim.Adam( params, lr =lr , weight_decay=5e-4)
        # training loop
        for k in range(0, steps):
        
            self.encoder.train()
            self.decoder.train()
            self.casual_decoder.eval()
            self.classifier.eval()
            acc_accum = 0
            num = 0
            if k < 50:
                for batch in data: 
                    z, mu, logvar= self.encoder(batch)
                    Xhat, adj = self.decoder(z)

                    alpha = z[:,:Nalpha]
                    print(alpha.shape)
                    beta = z[:,Nalpha:]   
                    print(beta.shape)     

                    nll = VAE_LL_loss(batch.x, Xhat, logvar, mu, self.device)
                    #compute matual information between alpha and beta
                    MI = calculate_MI(alpha, beta)
                    #compute casual effect
                    labels = torch.LongTensor(batch.y).to(self.device)
                    causalEffect,logits= joint_uncond(alpha,beta, batch, self.casual_decoder, self.classifier,self.device,k)
                    #compute gradient
                    alpha_loss = lam*nll + r * MI - ce*causalEffect

                    self.opt.zero_grad()
                    alpha_loss.backward()
                    self.opt.step()
                    print(alpha_loss.item())
            
            else:
                self.encoder.eval()
                self.decoder.eval()
                self.casual_decoder.train()
                self.classifier.train()
                acc_accum = 0
                num = 0
                for batch in data: 
                    z, mu, logvar= self.encoder(batch)
                    # #print(Xhat.shape)
                    alpha = z[:,:Nalpha]   
                    beta = z[:,Nalpha:] 
                    #compute matual information between alpha and beta
                    MI = calculate_MI(alpha, beta)
                    #compute conditional mutual information
                    causalEffect, logits= joint_uncond(alpha,beta, batch, self.casual_decoder,self.classifier,  self.device,k)
                    #compute gradient
                    labels = torch.LongTensor(batch.y).to(self.device)
                    loss = criterion(logits, labels) + r * MI - ce * causalEffect

                    print("classification loss: %f" %(criterion(logits, labels)))
                    # optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pred = logits.max(1, keepdim=True)[1]
                    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
                    acc = correct / float(len(batch.y))
                    acc_accum = acc_accum + acc
                    num = num + 1
                
                acc_train = acc_accum/num
                print("accuracy train: %f" %(acc_train))

                acc_eval = self.evaluate(eval_data, self.encoder, self.casual_decoder, self.classifier, Nalpha,batch_size, self.device,k)
                print("accuracy test: %f" %(acc_eval))

                acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test = self.test(test_data, self.encoder, self.casual_decoder, self.classifier, Nalpha,batch_size, self.device,k)
                print("accuracy test: %f" %(acc_test))

                filename="MUTAG.txt"
                if not os.path.exists(filename):
                    with open(filename, 'w') as f:
                        f.write("%f %f %f %f %f %f %f %f %f" % (loss, acc_train, acc_eval, acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test))
                        f.write("\n")
                else:
                    with open(filename, 'a+') as f:
                        f.write("%f %f %f %f %f %f %f %f %f" % (loss, acc_train, acc_eval, acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test))
                        f.write("\n")

        return loss

    def evaluate(self, data, encoder, casual_decoder, classifier, Nalpha, batch_size, device,k): 
        acc_accum = 0
        num = 0
        for batch in data: 
            z, mu, logvar= encoder(batch)

            alpha = z[:,:Nalpha]
            beta = z[:,Nalpha:]        

            labels = torch.LongTensor(batch.y).to(self.device)
            _ , logits = joint_uncond(alpha,beta, batch, casual_decoder, classifier, device,k)

            pred = logits.max(1, keepdim=True)[1]
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(batch.y))
            acc_accum = acc_accum + acc
            num = num + 1
        acc_test = acc_accum/num

        return acc_test

    def test(self, data, encoder, casual_decoder, classifier, Nalpha, batch_size, device,k): 
        acc_accum = 0
        sen_accum = 0
        spc_accum = 0
        prc_accum = 0
        f1s_accum = 0
        mcc_accum = 0
        num = 0
        for batch in data: 
            z, mu, logvar= encoder(batch)

            alpha = z[:,:Nalpha]
            beta = z[:,Nalpha:]        

            labels = torch.LongTensor(batch.y).to(self.device)
            _ , logits = joint_uncond(alpha,beta, batch, casual_decoder, classifier, device,k)

            pred = logits.max(1, keepdim=True)[1]
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()
            if len(labels)>1:
                test_acc, test_sen, test_spc, test_prc, test_f1s, test_mcc = self.calc_performance_statistics(pred,labels)
                acc = correct / float(len(batch.y))
                acc_accum = acc_accum + acc
                sen_accum = sen_accum + test_sen
                spc_accum = spc_accum + test_spc
                prc_accum = prc_accum + test_prc
                f1s_accum = f1s_accum + test_f1s
                mcc_accum = mcc_accum + test_mcc
                num = num + 1
        acc_test = acc_accum/num
        sen_test = sen_accum/num
        spc_test = spc_accum/num
        prc_test = prc_accum/num
        f1s_test = f1s_accum/num
        mcc_test = mcc_accum/num

        return acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test

    def calc_performance_statistics(self, y_pred, y):

        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
        N = TN + TP + FN + FP
        S = (TP + FN) / N
        P = (TP + FP) / N
        acc = (TN + TP) / N
        sen = TP / (TP + FN)
        spc = TN / (TN + FP)
        prc = TP / (TP + FP)
        f1s = 2 * (prc * sen) / (prc + sen)
        mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))

        return acc, sen, spc, prc, f1s, mcc