class LossBCE(Module):
    """implementation of the Binary Cross Entropy Loss"""
    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, y_pred, t):
        self.t = t
        self.y_pred = y_pred
        true_target, true_index = torch.max(t,1) # .view(1,-1)
        #print('index',true_index)

        y_true = (y_pred*t).sum(1)
        y_false = (y_pred*(1-t)).sum(1)
        #print('y_pred', y_pred)
        #print('our solution',(y_pred*t).sum(1))
        #print('y_pred[0]', y_pred[:,0])
        #print('yTrue',y_true)

        self.true_index = true_index

        self.y_true = y_true
        self.y_false = y_false
        # old way to do it
        L_true = -torch.log(self.y_true)
        L_false = -torch.log(1-self.y_false)
        pred_true_target = y_pred.gather(1,t[:,0].long().view(-1,1))
        #b.gather(1,a[:,0].long().view(-1,1))
        #L_true = -t[0]*torch.log(y_pred[0])-t[1]*torch.log(y_pred[1])
        #L_true = -(1-t[0])*torch.log(1-y_pred[0])-(1-t[1])*torch.log(1-y_pred[1])
        #L = -(  (t)*torch.log(y_pred) + (1-t)*torch.log(1-y_pred))
        #L =  - (  torch.log(pred_true_target) + torch.log(1-pred_true_target))
        #L = (  (t[0])*torch.log(1-y_pred) + (t)*torch.log(y_pred))
        #print(L)
        #L = L.sum(1)
        L = torch.cat((L_true,L_false),0)
        #print(L.size())
        #print('L',torch.mean(L,0))
        return torch.mean(L)

    def forward2(self, y_pred, t):
        for i in range(len(t)):
            if(t[i,0]==1):
                l[i] = - torch.log(Tensor(y_pred[i,0]))
            else:
                l[i] = - torch.log(Tensor(1-y_pred[i,0]))
        return torch.mean(l)

    def backward(self):
        y_true =  (self.y_pred*self.t).sum(1)
        y_false = (self.y_pred*(1-self.t)).sum(1)
        # old way
        dL_true = -(self.t).sum(1)/ (self.y_pred*self.t).sum(1)
        dL_false = (1-self.t).sum(1) /(1- (self.y_pred*(1-self.t)).sum(1) )
        dL = torch.cat((dL_true,dL_false),0)
        #dL = -(self.true_target*1/self.y_pred + (1-self.true_target)*1/(1-self.y_pred))
        return torch.mean(dL)

    def param(self):
        return self.params
