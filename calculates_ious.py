
import numpy as np
import torch
def calculates_ious(boxes_preds:torch.tensor, boxes_labels:torch.tensor, box_format="midpoint",eps=1e-6):
    '''
    计算目标检测任务时的IOU
    Args:
        boxes_preds:   [batch_size, S_cells, S_cells, M ,4 ]，或者[M ,4 ]——【(*,M, 4)】
        boxes_labels:  [batch_size, S_cells, S_cells, N ,4 ]，或者[N ,4 ]——【(*,N, 4)】
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        eps: 一个极小的数， 防止分母为0
    Returns: [batch_size, S_cells, S_cells, M ,N]，或者[M ,N]——【(*,M,N)】
    '''
    
    boxes_preds_shape = boxes_preds.shape
    boxes_labels_shape = boxes_labels.shape

    ndims_boxes_preds = len(boxes_preds_shape)
    ndims_boxes_labels = len(boxes_labels_shape)

    assert ndims_boxes_preds == ndims_boxes_labels, "boxes_preds与boxes_labels的维度数必须相等"
    assert boxes_preds_shape[-1] == boxes_labels_shape[-1] == 4, "boxes_preds与boxes_labels的最后一个维度必须等于4"

    prefix_size = boxes_preds_shape[:-2]
    M_origin = boxes_preds_shape[-2]
    N_origin = boxes_labels_shape[-2]

    expand_size = prefix_size + (M_origin, N_origin, 2)

    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (*,M, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (*,N, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        # [Xc,Yc,W,H]
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        boxes_preds[...,0:1] = box1_x1
        boxes_preds[...,1:2] = box1_y1
        boxes_preds[...,2:3] = box1_x2
        boxes_preds[...,3:4] = box1_y2

        boxes_labels[...,0:1] = box2_x1
        boxes_labels[...,1:2] = box2_y1
        boxes_labels[...,2:3] = box2_x2
        boxes_labels[...,3:4] = box2_y2

        # boxes_preds_new = torch.zero_(boxes_preds)
        # boxes_preds_new[...,0:1] = box1_x1
        # boxes_preds_new[...,1:2] = box1_y1
        # boxes_preds_new[...,2:3] = box1_x2
        # boxes_preds_new[...,3:4] = box1_y2
        #
        # boxes_labels_new = torch.zero_(boxes_labels)
        # boxes_labels_new[...,0:1] = box2_x1
        # boxes_labels_new[...,1:2] = box2_y1
        # boxes_labels_new[...,2:3] = box2_x2
        # boxes_labels_new[...,3:4] = box2_y2

    if box_format == "corners": # (x1,y1,x2,y2)
        pass
        # boxes_preds_new = boxes_preds
        # boxes_labels_new = boxes_labels
    # boxes_preds = boxes_preds_new[..., 2:].reshape(prefix_size+(M_origin,2))
    boxes_preds_temp = boxes_preds[..., 2:]
    boxes_preds_unsqueeze = boxes_preds_temp.unsqueeze(-2)

    boxes_labels_temp = boxes_labels[..., 2:].reshape(prefix_size + (N_origin, 2))
    boxes_labels_unsqueeze = boxes_labels_temp.unsqueeze(-3)

    max_xy = torch.min(boxes_preds_unsqueeze.expand(expand_size),
                       boxes_labels_unsqueeze.expand(expand_size))
    min_xy = torch.max(boxes_preds[..., :2].reshape(prefix_size + (M_origin, 2)).unsqueeze(-2).expand(expand_size),
                       boxes_labels[..., :2].reshape(prefix_size + (N_origin, 2)).unsqueeze(-3).expand(expand_size))

    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[..., 0] * inter[..., 1]
    # 计算先验框和真实框各自的面积
    area_a = ((boxes_preds[..., 2] - boxes_preds[..., 0]) * (boxes_preds[..., 3] - boxes_preds[..., 1])).reshape(
        prefix_size + (M_origin,)).unsqueeze(-1).expand_as(inter)  # [M,N]
    area_b = ((boxes_labels[..., 2] - boxes_labels[..., 0]) * (boxes_labels[..., 3] - boxes_labels[..., 1])).reshape(
        prefix_size + (N_origin,)).unsqueeze(-2).expand_as(inter)  # [M,N]
    union = area_a + area_b - inter
    iou_with_adjust = inter / (union + eps)

    return iou_with_adjust

def xywh(bboxA, bboxB):
    """
    This function computes the intersection over union between two
    2-d boxes (usually bounding boxes in an image.
    Attributes:
        bboxA (list): defined by 4 values: [xmin, ymin, width, height].
        bboxB (list): defined by 4 values: [xmin, ymin, width, height].
        (Order is irrelevant).
    Returns:
        IOU (float): a value between 0-1 representing how much these boxes overlap.
    """

    xA, yA, wA, hA = bboxA
    areaA = wA * hA
    xB, yB, wB, hB = bboxB
    areaB = wB * hB

    overlap_xmin = max(xA-wA/2, xB-wB/2)
    overlap_ymin = max(yA-hA/2, yB-wB/2)
    overlap_xmax = min(xA + wA/2, xB + wB/2)
    overlap_ymax = min(yA + hA/2, yB + hB/2)

    W = overlap_xmax - overlap_xmin
    H = overlap_ymax - overlap_ymin

    if min(W, H) < 0:
        return 0

    intersect = W * H
    union = areaA + areaB - intersect

    return intersect / union

def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0
    x1max, y1max = box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0
    x2max, y2max = box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou
if __name__ == '__main__':
    a = torch.tensor([100,50,200,100])
    b = torch.tensor([100, 100, 200, 200])
    c = torch.tensor([110., 100., 200., 200.])


    a2 = torch.tensor([80,50,200,100])
    b2 = torch.tensor([50, 100, 200, 200])
    c2 = torch.tensor([120., 120., 220., 220.])

    A = [a,b,c]
    B = [a2,b2,c2]
    for i in range(3):
        for j in range(3):
            print("1 xywh:",xywh(A[i], B[j]))
            print("2 box_iou_xywh:",box_iou_xywh(A[i], B[j]))

    print(calculates_ious(torch.stack((a,b,c), dim=0), torch.stack((a2,b2,c2), dim=0), eps=0))
    print(calculates_ious(torch.stack((a, b), dim=0), torch.stack((a2, b2, c2), dim=0), eps=0))


    print(calculates_ious(torch.stack((a, b, c), dim=0), torch.stack((a2, b2), dim=0), eps=0))
    At1=torch.stack((a, b, c))
    At2= torch.stack((a2, b2, c2))
    print(calculates_ious(torch.stack((At1,At2), dim=0), torch.stack((At2,At1), dim=0), eps=0))
