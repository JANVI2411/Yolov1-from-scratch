import torch
import torch.nn as nn

class YoloLoss(nn.Module):
  def __init__(self,S=7 ,B=2, C=20):
    super(YoloLoss,self).__init__()
    self.mse = nn.MSELoss(reduction="sum")
    self.lambda_coord = 5
    self.lambda_noobj = 0.5
    self.S = S
    self.B = B
    self.C = C

  # predictions = (c0 to c19,p20,x21,y22,h23,w24,p25,x26,y27,h28,w29) Tensor[0 to 29]
  # target = (c0 to c19,p20,x21,y22,h23,w24) Tensor[0 to 24]
  def forward(self, predictions, target):
    predictions = predictions.reshape(-1,self.S,self.S,self.C + (self.B*5))
    iou_b1 = intersection_over_union(predictions[...,21:25],target[...,21:25]) # return [N,S,S,1]
    iou_b2 = intersection_over_union(predictions[...,26:30],target[...,21:25]) # return [N,S,S,1]
    iou_combine = torch.cat([iou_b1.unsqueeze(0),iou_b1.unsqueeze(0)],dim=0) # return [1,N,S,S,1]
    iou_score, best_iou_idx = torch.max(iou_combine,dim=0) ## (N,S,S,1), (N,S,S,1)
    exists_box = target[...,20].unsqueeze(-1) # (N,S,S,1)
    
    ### box coordinate -x,y,w,h###
    prediction_boxes = exists_box*(best_iou_idx*predictions[...,26:30] + (1-best_iou_idx)*predictions[...,21:25]) # (N,S,S,4)
    target_boxes = exists_box*target[...,21:25] # (N,S,S,4) 
    sign = torch.sign(prediction_boxes[...,2:4])
    prediction_boxes[...,2:4] = sign* torch.sqrt(torch.abs(prediction_boxes[...,2:4]+1e-6)) # # (N,S,S,4)
    target_boxes[...,2:4] = torch.sqrt(target_boxes[...,2:4]) # (N,S,S,4)
    prediction_boxes = torch.flatten(prediction_boxes,end_dim=-2) # (N*S*S,4)
    target_boxes = torch.flatten(target_boxes,end_dim=-2) # (N*S*S,4)
    box_coord_loss = self.mse(prediction_boxes,target_boxes) # (1,4) -> reduction_sum -> (1)
    
    ### loss obj class prob - p ###
    prediction_obj = best_iou_idx*predictions[...,25:26] + (1-best_iou_idx)*predictions[...,20:21] # (N,S,S,1)
    box_obj_loss = self.mse(torch.flatten(exists_box*prediction_obj),
                            torch.flatten(exists_box*target[...,20:21])) # (1)

    ### loss no-obj class prob - p ###
    box_no_obj_loss = self.mse(torch.flatten((1-exists_box)*predictions[...,25:26]),
                               torch.flatten((1-exists_box)*target[...,20:21]))
    box_no_obj_loss += self.mse(torch.flatten((1-exists_box)*predictions[...,20:21]),
                               torch.flatten((1-exists_box)*target[...,20:21]))

    ### loss all class prob - c###
    predictions_class = torch.flatten(exists_box*predictions[...,:20],end_dim=-2) # (N*S*S,20) 
    target_class = torch.flatten(exists_box*target[...,:20],end_dim=-2) # (N*S*S,20) 
    box_class_loss = self.mse(predictions_class,target_class) # (N*S*S,1) -> reduction_sum -> (1)

    ## final loss ##
    # print(box_coord_loss.shape,box_obj_loss.shape,box_no_obj_loss.shape,box_class_loss.shape)
    # print(box_coord_loss,box_obj_loss,box_no_obj_loss,box_class_loss)
    loss = (self.lambda_coord*box_coord_loss) + box_obj_loss + (self.lambda_noobj*box_no_obj_loss) + box_class_loss
    return loss