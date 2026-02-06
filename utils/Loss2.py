import torch
import torch.nn as nn
import torch.nn.functional as F
from visualization import *
from torch_topological.nn import WassersteinDistance,CubicalComplex
#from skimage.feature import local_binary_pattern 
#import gudhi as gd
#from gudhi.wasserstein import wasserstein_distance
#import gudhi as gd


class DINOLoss(nn.Module):
    def __init__(self, out_dim=512, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs,teacher_temp):
        
        total_loss, n_loss_terms = 0.0, 0

        # Teacher softmax with centering & temperature
        teacher_logits = [(t - self.center.to(t.device)) / teacher_temp for t in teacher_outputs]
        teacher_probs = [F.softmax(logits, dim=-1).detach() for logits in teacher_logits]

        # Student scaled logits
        student_logits = [s / self.student_temp for s in student_outputs]

        for t_out in teacher_probs:
            for s_log in student_logits:
                loss = torch.sum(-t_out * F.log_softmax(s_log, dim=-1), dim=-1).mean()  # shape: (B,)
                total_loss += loss
                n_loss_terms += 1
        # Update center (only from teacher views)
        batch_center = torch.cat(teacher_outputs, dim=0).mean(dim=0, keepdim=True)
        with torch.no_grad():
            self.center = self.center.to(batch_center.device) * self.center_momentum + \
                        (1 - self.center_momentum) * batch_center

        return total_loss / n_loss_terms

class Topological_Loss(torch.nn.Module):

    def __init__(self, lam=0.1):
        super().__init__()
        self.lam                = lam
        #self.vr                 = VietorisRipsComplex(dim=self.dimension)
        self.cubicalcomplex     = CubicalComplex()
        self.wloss              = WassersteinDistance(p=2)
        self.sigmoid_f          = nn.Sigmoid()
        self.avgpool            = nn.AvgPool2d(2,2)
  
    def forward(self, model_output,labels):

        totalloss             = 0
        model_output_r        = self.avgpool(self.avgpool(self.avgpool(model_output)))
        labels_r              = self.avgpool(self.avgpool(self.avgpool(labels)))
        model_output_r        = self.sigmoid_f(model_output_r)
        predictions           = torch.squeeze(model_output_r,dim=1) 
        masks                 = torch.squeeze(labels_r,dim=1)
        pi_pred               = self.cubicalcomplex(predictions)
        pi_mask               = self.cubicalcomplex(masks)
        
        for i in range(predictions.shape[0]):

            topo_loss   = self.wloss(pi_mask[i],pi_pred[i])             
            totalloss   +=topo_loss
        loss             = self.lam * totalloss/predictions.shape[0]
        return loss

class Dice_CE_Loss():
    def __init__(self):

#        self.batch,self.h,self.w,self.n_class = inputs.shape

        self.sigmoid_f     = nn.Sigmoid()
        self.softmax       = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.bcewithlogic = nn.BCEWithLogitsLoss(reduction="mean")
    
    def Dice_Loss(self,input,target):

        smooth          = 1
        input           = self.sigmoid_f(torch.flatten(input=input))
        target          = torch.flatten(input=target)
        intersection    = (input * target).sum()
        dice_loss       = 1- (2.*intersection + smooth )/(input.sum() + target.sum() + smooth)
        return dice_loss

    def BCE_loss(self,input,target):
        input           = torch.flatten(input=input)
        target          = torch.flatten(input=target)
        sigmoid_f       = nn.Sigmoid()
        sigmoid_input   = sigmoid_f(input)
        #B_Cross_Entropy = F.binary_cross_entropy(sigmoid_input,target)
        entropy_with_logic = self.bcewithlogic(input,target)
        return entropy_with_logic

    def Dice_BCE_Loss(self,input,target):
        return self.Dice_Loss(input,target) + self.BCE_loss(input,target) 
    
    
    # Manuel cross entropy loss 
    def softmax_manuel(self,input):
        return (torch.exp(input).t() / torch.sum(torch.exp(input),dim=1)).t()

    def CE_loss_manuel(self, input,target):

        last_dim = torch.tensor(input.shape[:-1])
        last_dim = torch.prod(last_dim)
        input    = input.reshape(last_dim,-1)      
        target   = target.view(last_dim,-1)     #    should be converted one hot previously

        return torch.mean(-torch.sum(torch.log(self.softmax_manuel(input)) * (target),dim=1))


    # CE loss 
    def CE_loss(self,input,target):
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        last_dim = torch.tensor(input.shape[:-1])
        last_dim = torch.prod(last_dim)
        input    = input.reshape(last_dim,-1)
        target   = target.reshape(last_dim).long         #  it will be converted one hot encode in nn.CrossEnt 

        return cross_entropy(input,target)
