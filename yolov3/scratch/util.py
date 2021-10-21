from logging import exception
import torch
import numpy as np
from mylogger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

np.set_printoptions(precision=3, suppress=True)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        # box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        # upper_left_x = box[0] - box[2] / 2
        # upper_left_y = box[1] - box[3] / 2
        upper_left_x = box[0]
        upper_left_y = box[1]
        w_box = box[2]-upper_left_x
        h_box = box[3]-upper_left_y

        width = height =1
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            w_box * width,
            h_box * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = torch.cuda.is_available()):
    
    if CUDA:
        prediction=prediction.cuda()
    batch_size = prediction.size(0)
    feature_map_dim = prediction.size(2)
    stride =  inp_dim // feature_map_dim
    grid_size = inp_dim // stride
    logger.warning(f'inp_dim={inp_dim},feature_map_dim={feature_map_dim},stride={stride}, grid_size= {grid_size}')
    
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    logger.warning(f'grid_size={grid_size},num_anchors={num_anchors},total num_anchors (grid_size*grid_size*num_anchors) = {grid_size*grid_size*num_anchors}')
    # logger.info(f'predict_transform的输出shape为{prediction.shape}')
    # 锚的尺寸根据net块的height和width属性。这些属性是输入图像的尺寸，它比检测图大（输入图像是检测图的stride倍）。
    # 因此，我们必须通过检测特征图的stride来划分锚。
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidence，控制在[0,1]之间
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) #centre_X
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) #centre_Y
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) #object Confidence
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors #宽、高
    # 类别logits分数使用sigmoid激活函数，而不是softmax
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride #各个stride再乘回来，将检测图调整为输入图像的大小，就统一到同一个尺度了
    
    return prediction

def calculates_ious(boxes_preds: torch.tensor, boxes_labels: torch.tensor, box_format="midpoint", eps=1e-6):
    # assert (boxes_labels < 0).sum() == 0
    # todo 小心检查：这里会改变值
    '''
    计算目标检测任务时的IOU
    Args:
        boxes_preds(torch.tensor):   [batch_size, S_cells, S_cells, M ,4 ]，或者[M ,4 ]
        boxes_labels(torch.tensor):  [batch_size, S_cells, S_cells, N ,4 ]，或者[N ,4 ]
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        eps: 一个极小的数， 防止分母为0
    Returns(torch.tensor): [batch_size, S_cells, S_cells, M ,N]，或者[M ,N]
    UnitTest : yolos/yolov1/utils/utils_func_test.py
    '''

    boxes_preds_shape = boxes_preds.shape
    boxes_labels_shape = boxes_labels.shape

    ndims_boxes_preds = len(boxes_preds_shape)
    ndims_boxes_labels = len(boxes_labels_shape)
    if ndims_boxes_preds<ndims_boxes_labels:
        boxes_preds = boxes_preds.unsqueeze(dim=0)
    boxes_preds_shape = boxes_preds.shape
    ndims_boxes_preds = len(boxes_preds_shape)
    assert ndims_boxes_preds == ndims_boxes_labels, "boxes_preds与boxes_labels的维度数必须相等"
    assert boxes_preds_shape[-1] == boxes_labels_shape[-1] == 4, "boxes_preds与boxes_labels的最后一个维度必须等于4"

    prefix_size = boxes_preds_shape[:-2]
    M_origin = boxes_preds_shape[-2]
    N_origin = boxes_labels_shape[-2]

    expand_size = prefix_size + (M_origin, N_origin, 2)

    # todo 这里使用新的tesor是防止改变值，会改变值
    boxes_preds_new = copy.deepcopy(boxes_preds)
    boxes_labels_new = copy.deepcopy(boxes_labels)
    # boxes_preds_new = torch.zeros_like(boxes_preds)
    # boxes_labels_new = torch.zeros_like(boxes_labels)

    if box_format == "midpoint":
        #输入为 [Xc,Yc,W,H]
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        boxes_preds_new[..., 0:1] = box1_x1
        boxes_preds_new[..., 1:2] = box1_y1
        boxes_preds_new[..., 2:3] = box1_x2
        boxes_preds_new[..., 3:4] = box1_y2

        boxes_labels_new[..., 0:1] = box2_x1
        boxes_labels_new[..., 1:2] = box2_y1
        boxes_labels_new[..., 2:3] = box2_x2
        boxes_labels_new[..., 3:4] = box2_y2
    if box_format == "corners":  # (x1,y1,x2,y2)
        # logger.error(f"===boxes_preds_new={boxes_preds_new}")
        # logger.error(f"===boxes_preds_new={boxes_labels_new}")
        pass
    # 使用temp变量是为了逐步展示怎么得到 boxes_preds_unsqueeze
    # （x2，y2）
    boxes_preds_temp = boxes_preds_new[..., 2:]
    boxes_preds_unsqueeze = boxes_preds_temp.unsqueeze(-2)
    boxes_labels_temp = boxes_labels_new[..., 2:].reshape(prefix_size + (N_origin, 2))
    boxes_labels_unsqueeze = boxes_labels_temp.unsqueeze(-3)
    max_xy = torch.min(boxes_preds_unsqueeze.expand(expand_size),boxes_labels_unsqueeze.expand(expand_size))
    # （x1，y1）
    min_xy = torch.max(boxes_preds_new[..., :2].reshape(prefix_size + (M_origin, 2)).unsqueeze(-2).expand(expand_size),
                       boxes_labels_new[..., :2].reshape(prefix_size + (N_origin, 2)).unsqueeze(-3).expand(expand_size))

    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[..., 0] * inter[..., 1]
    # 计算先验框和真实框各自的面积
    area_a = ((boxes_preds_new[..., 2] - boxes_preds_new[..., 0]) * (boxes_preds_new[..., 3] - boxes_preds_new[..., 1])).reshape(
        prefix_size + (M_origin,)).unsqueeze(-1).expand_as(inter)  # [M,N]
    area_b = ((boxes_labels_new[..., 2] - boxes_labels_new[..., 0]) * (boxes_labels_new[..., 3] - boxes_labels_new[..., 1])).reshape(
        prefix_size + (N_origin,)).unsqueeze(-2).expand_as(inter)  # [M,N]
    union = area_a + area_b - inter
    union[union == 0] = eps
    iou_with_adjust = inter / union
    # assert (iou_with_adjust < 0).sum() == 0
    return iou_with_adjust

def get_test_ious():
    a = torch.tensor([100, 50, 200, 100])
    b = torch.tensor([100, 100, 200, 200])
    c = torch.tensor([110., 100., 200., 200.])

    a2 = torch.tensor([80, 50, 200, 100])
    b2 = torch.tensor([50, 100, 200, 200])
    c2 = torch.tensor([120., 120., 220., 220.])

    At1=torch.stack((a, b, c))
    At2= torch.stack((a2, b2, c2))

    ious = calculates_ious(a,At2)
    ious2 = bbox_iou(a.unsqueeze(0),At2)
    return ious,ious2

def non_max_suppression(predictions_logits:torch.tensor, num_classes:int=80, conf_thres:float=0.4, nms_thres:float = 0.4,
                        use_soft_nms=True,soft_nms_sigma=0.5,soft_conf_thres=0.4):
    predictions = copy.deepcopy(predictions_logits)
    # logger.error(f"变换前{predictions[0, 0, 0:4]}")
    bs = predictions.size(0)
    # 将框转换成左上角右下角的形式
    # todo shape_boxes = torch.zeros_like(predictions[:,:,:4])
    shape_boxes = predictions.new(predictions[:,:,:4].shape)
    shape_boxes[:,:,0] = predictions[:,:,0] - predictions[:,:,2]/2 #x0_tl
    shape_boxes[:,:,1] = predictions[:,:,1] - predictions[:,:,3]/2 #y0_tl
    shape_boxes[:,:,2] = predictions[:,:,0] + predictions[:,:,2]/2 #x1_br
    shape_boxes[:,:,3] = predictions[:,:,1] + predictions[:,:,3]/2 #y1_br

    predictions[:,:,:4] = shape_boxes
    # logger.error(f"变换前{predictions[0, 0, 0:4]}")
    masks = predictions[..., 4] >= conf_thres

    _total_detections = masks.sum()
    logger.info(f"  ---------总共{masks.numel()}个预测框，其中只有{_total_detections}个 >=conf_thres({conf_thres}),"
                f"  ---------后面还需通过【NMS】 <=nms_thres({nms_thres})来过滤！"
                f"  ---------   或者通过【NMS_Soft】")
    if _total_detections == 0:
        return None

    output = []
    # 1、对所有图片进行循环。
    for i in range(bs):
        # 2、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
        mask = masks[i]
        if mask.sum() ==0:
            continue
        prediction = predictions[i]
        prediction = prediction[mask]

        # 3、判断第2步中获得的框的种类与得分。
            # 取出预测结果中框的位置与之进行堆叠。
            # 此时最后一维度里面的内容由5+num_classes变成了4+1+2，
            # 四个参数代表框的位置，一个参数代表预测框是否包含物体obj_conf，两个参数分别代表种类的置信度cls_conf与种类。
        '''
                # output = torch.max(input, dim) 
                import torch
                a = torch.tensor([[1,5,62,54], 
                                [2,6,2,6], 
                                [2,65,2,6]])
                print(a.shape) # torch.Size([3, 4])

                out_0_column = torch.max(a, 0) #0是每列的最大值
                print(out_0_column)
                # torch.return_types.max(
                # values=tensor([ 2, 65, 62, 54]),
                # indices=tensor([1, 2, 0, 0]))
                print("==="*10)
                out_1_row = torch.max(a, 1) #1是每行的最大值
                print(out_1_row)
                # torch.return_types.max(
                # values=tensor([62,  6, 65]),
                # indices=tensor([2, 3, 1]))
        '''
        max_conf_values, max_conf_indices = torch.max(prediction[:,5:5+ num_classes], 1)
        # if max_conf_values.
        total_classes = max_conf_indices.tolist()
        if len(total_classes) == 0:
            continue
        unique_classes = list(set(total_classes))  # 得到unique类别

        logger.info(f"......图片{i}中的全部类别{len(total_classes)}个：{total_classes}")
        logger.info(f"......图片{i}中的独立类别{len(unique_classes)}个：{unique_classes}")

        # shape: [total_anchors_num,] -->[total_anchors_num,1]
        max_conf_values = max_conf_values.float().unsqueeze(1)
        # shape: [total_anchors_num,] -->[total_anchors_num,1]
        max_conf_indices = max_conf_indices.float().unsqueeze(1)
        seq = (prediction[:,:5], max_conf_values, max_conf_indices)
        detections = torch.cat(seq, 1) # [total_anchors_num,7]

        best_box = []
        # 4、对种类进行循环，
            # 非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
            # 对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。

        for c in unique_classes:
            cls_mask = detections[:,-1] == c

            detection = detections[cls_mask]
            conf_scores = detection[:,4]
            # 5、根据得分对该种类进行从大到小排序。
            conf_scores_sort_index = torch.argsort(conf_scores, descending=True)
            detection = detection[conf_scores_sort_index]
            detection_in_cls = detection.size(0)  # Number of detections
            logger.info(f"         图片{i}中类别{c}对应的预测框的个数{detection_in_cls}")
            # logger.info(f"detection\n{detection.cpu().numpy()}")
            while detection.size(0)>0:
                # 6、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                best_box.append(detection[0])
                if detection.size(0) == 1:
                    break
                concurrent_box = best_box[-1][0:4]
                other_boxes = detection[1:,0:4]
                ious = calculates_ious(concurrent_box,other_boxes,box_format = "corners")
                # 测试两种iou计算方法的一致性，已通过
                # ious2 = bbox_iou(concurrent_box.unsqueeze(0), other_boxes)
                # print(ious,ious2)
                # assert ious.sum()-ious2.sum()==0
                logger.error(f"calculates_ious={ious}")
                # ious = get_test_ious() #todo 测试
                ious = ious[0]
                if use_soft_nms: #不直接过滤，而是根据iou计算一个调整系数，然后重新排序
                    assert soft_nms_sigma>0 and soft_conf_thres>0
                    detection = detection[1:] #只关注后面的
                    _adjust = torch.exp(-(ious*ious)/soft_nms_sigma) #将获得的IOU取高斯指数后乘上原得分，之后重新排序
                    detection[:,4] = _adjust * detection[:,4]
                    mask_soft_conf = detection[:,4] >= soft_conf_thres
                    detection = detection[mask_soft_conf]
                    if detection.size(0)>1:
                        conf_scores = detection[:, 4]
                        # 5、根据【调整后的】得分对该种类进行从大到小排序。
                        conf_scores_sort_index = torch.argsort(conf_scores, descending=True)
                        detection = detection[conf_scores_sort_index]


                else:# 小于iou阈值的才保留，否则就过滤掉了
                    keeped_idx = ious<nms_thres
                    '''
                    #todo 测试(与上面的get_test_ious()配合的)
                    if keeped_idx.sum()<len(ious):
                        _to_be_same_len = detection[1:].size(0)-len(ious)
                        if _to_be_same_len>0:
                            _need_added = torch.tensor([False]*_to_be_same_len)
                            keeped_idx = torch.cat((keeped_idx,_need_added))
                        elif _to_be_same_len==0:
                            pass
                        else:
                            keeped_idx = keeped_idx[0:_to_be_same_len]
                    '''
                    _num_filered = len(other_boxes) - keeped_idx.sum()
                    logger.info(f"      ---------图片{i}中类别{c} 共{detection.size(0)}个，Others {len(other_boxes)}个，过滤了{_num_filered}")
                    _total_detections-=_num_filered
                    logger.info(f"    =========总共剩余{_total_detections}个")
                    detection = detection[1:][keeped_idx]

        best_box = torch.stack(best_box, dim=0)
        idx_batch = best_box.new(best_box.size(0), 1).fill_(i)
        best_box = torch.cat((best_box,idx_batch), 1)
        logger.info(f"-----------最终留下的个数{best_box.size(0)}")
        #[num_boxes,8]
        # 其中8位的含义是4个坐标(x,y,x,y),1个obj_conf,1个cls_conf,1个类别id,1个idx_batch
        output.append(best_box)
    output = torch.cat(output)
    # [num_boxes,8]
    # 其中8位的含义是4个坐标(x,y,x,y),1个obj_conf,1个cls_conf,1个类别id,1个idx_batch
    return output #[Tensor(num_boxes,8) or None]
    # 函数在上面还有一个分支，可能会返回None


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1,min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def bbox_iou_1(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def write_results(prediction:torch.tensor, confidence:float=0.05,num_classes:int=80, nms_conf:float = 0.3):
    '''
    prediction: torch.tensor, shape(batch_size, total_num_anchors, 85)
                             含义为[batch_size,  total_num_anchors,   5(cX,cY,w,h,confidence) + 80(num_classes)]
    '''
    # logger.error(f"变换前{prediction[0, 0, 0:4]}")
    conf_mask = (prediction[:,:,4]>confidence).float() # shape (batch_size, total_anchors_num)
    conf_mask = conf_mask.unsqueeze(2)                 # shape (batch_size, total_anchors_num , 1)
    prediction = prediction*conf_mask                  # shape (batch_size, total_anchors_num, 85)， 没变
    # logger.info(prediction.shape)

    box_corner = prediction.new(prediction.shape)
    #  (cX,cY,w,h)--> (tlrb)
    #  (center x, center y, height, width)  to 
    #  (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    # 从box_corner 将转换后的值赋值给 prediction
    # 现在prediction[:,:,:4]的值为(top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    prediction[:,:,:4] = box_corner[:,:,:4]
    # logger.error(f"变换后{prediction[0, 0, 0:4]}")

    batch_size = prediction.size(0)
    write = False
    num_prediction_reserved_in_batch = 0
    for idx in range(batch_size):# 一个batch中的图片依次循环
        num_prediction_reserved_in_image = 0
        image_pred = prediction[idx] #shape  (total_anchors_num, 85)
        #confidence threshholding 
        #NMS
        '''
                # output = torch.max(input, dim) 
                import torch
                a = torch.tensor([[1,5,62,54], 
                                [2,6,2,6], 
                                [2,65,2,6]])
                print(a.shape) # torch.Size([3, 4])

                out_0_column = torch.max(a, 0) #0是每列的最大值
                print(out_0_column)
                # torch.return_types.max(
                # values=tensor([ 2, 65, 62, 54]),
                # indices=tensor([1, 2, 0, 0]))
                print("==="*10)
                out_1_row = torch.max(a, 1) #1是每行的最大值
                print(out_1_row)
                # torch.return_types.max(
                # values=tensor([62,  6, 65]),
                # indices=tensor([2, 3, 1]))
        '''
        max_conf_values, max_conf_indices = torch.max(image_pred[:,5:5+ num_classes], 1)
        # shape: [total_anchors_num] -->[total_anchors_num,1]
        max_conf_values = max_conf_values.float().unsqueeze(1)
        # shape: [total_anchors_num] -->[total_anchors_num,1]
        max_conf_indices = max_conf_indices.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf_values, max_conf_indices)
        image_pred = torch.cat(seq, 1) # [total_anchors_num,7]
        # non_zero_idx的shape [非零个数,1]
        non_zero_idx =  (torch.nonzero(image_pred[:,4])) #第4个值confidence是非零的
        # non_zero_idx的shape [非零个数]
        non_zero_idx = non_zero_idx.squeeze() #所有非零的idx值

        try:
            l=len(non_zero_idx)
        except:
            l=0



        logger.info(f"图片{idx},非零non_zero_idx={non_zero_idx}，个数为{l}")
              
        try:
            image_pred_ = image_pred[non_zero_idx,:].view(-1,7) ## image_pred_的shape [非零个数,7]
        except:
            continue # handle situations where we get no detections
        #For PyTorch 0.4 compatibility
        #Since the above code with not raise exception for no detection 
        #as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue 
        
        #Get the various classes detected in the image
        # img_classes:shape[不同类别数量]
        img_classes = torch.unique(image_pred_[:,-1].detach().data)  # -1 index holds the class index
        logger.info(f"......全部类别：{len(image_pred_[:,-1].detach().data),image_pred_[:,-1].detach().data}")
        logger.info(f"......独立类别：{len(img_classes),img_classes}")

        
        for cls in img_classes:
            #perform NMS
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1) # 获取所有该类别（==cls）的预测值
            class_mask_idx = torch.nonzero(cls_mask[:,-2]).squeeze() # max_index   # 获取所有该类别（==cls）的预测中最大值为非零的indices
            image_pred_class = image_pred_[class_mask_idx].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top,按confidence从大到小排序
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx_in_cls = image_pred_class.size(0)   #Number of detections
            logger.info(f"         类别{cls}对应的预测框的个数{idx_in_cls}")
            
            # 应该是从类别{cls}对应的预测框个数  减去  iou重合大于nms_conf的个数
            num_prediction_reserved_in_cls = idx_in_cls
            for i in range(idx_in_cls):                

                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    # logger.error(f"===ious={ious},{image_pred_class[i].unsqueeze(0)[:,0:4]}")
                    # logger.error(f"===ious={ious},{image_pred_class[i + 1:][:, 0:4]},mode=corners")
                    logger.error(f"bbox_ious={ious}")
                except ValueError:
                    break            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                other_prediction_reserved = iou_mask.sum()
                num_deleted = iou_mask.numel()-other_prediction_reserved
                logger.info(f"              类别{cls}中box{i}与其后面{len(image_pred_class[i+1:])}个boxes的ious={ious},阈值为<{nms_conf}，小于阈值的目标数{other_prediction_reserved}")
                num_prediction_reserved_in_cls -= num_deleted
                
                logger.info(f"              类别{cls}中应删除的目标数为{num_deleted},剩余的目标数{num_prediction_reserved_in_cls}")
                
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_idx = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_idx].view(-1,7)
            


            batch_idx = image_pred_class.new(image_pred_class.size(0), 1).fill_(idx)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_idx, image_pred_class

            num_prediction_reserved_in_image += num_prediction_reserved_in_cls
            
            
            if not write:
                output = torch.cat(seq,1)
                logger.info(f"              图片{idx}-类别{cls}-output（write=False）.shape为{output.shape}")
                logger.info(f"            图片{idx}中累计保留的目标数{num_prediction_reserved_in_image}")
                write = True
            else:
                out = torch.cat(seq,1)
                logger.info(f"              图片{idx}-类别{cls}-的out.shape为{out.shape}")
                logger.info(f"            图片{idx}中累计保留的目标数{num_prediction_reserved_in_image}")
                output = torch.cat((output,out))
                logger.info(f"            整个Batch累计的-output（write=True）.shape为{output.shape}")


        num_prediction_reserved_in_batch+=num_prediction_reserved_in_image
    logger.info(f"共计保留的目标数{num_prediction_reserved_in_batch}")
    try:
        return output
    except:
        return 0


if __name__ == '__main__':
    # get_test_ious()
    # print(get_test_ious())
    # a = torch.tensor([100, 50, 200, 100])
    # b = torch.tensor([100, 100, 200, 200])
    # c = torch.tensor([110., 100., 200., 200.])
    #
    # a2 = torch.tensor([80, 50, 200, 100])
    # b2 = torch.tensor([50, 100, 200, 200])
    # c2 = torch.tensor([120., 120., 220., 220.])
    #
    # At1=torch.stack((a, b, c))
    # At2= torch.stack((a2, b2, c2))
    #
    # ious = calculates_ious(a,At2)

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from collections import Counter


    def intersection_over_union_back_v1(boxes_preds: torch.tensor, boxes_labels: torch.tensor, box_format="midpoint"):
        boxes_preds_shape = boxes_preds.shape
        boxes_labels_shape = boxes_labels.shape

        ndims_boxes_preds = len(boxes_preds_shape)
        ndims_boxes_labels = len(boxes_labels_shape)

        assert ndims_boxes_preds == ndims_boxes_labels, "boxes_preds与boxes_labels的维度数必须相等"
        assert boxes_preds_shape[-1] == boxes_labels_shape[-1] == 4, "boxes_preds与boxes_labels的最后一个维度必须等于4"

        M_origin = boxes_preds_shape[-2]
        N_origin = boxes_labels_shape[-2]

        prefix_size = ()
        if ndims_boxes_preds > 2:
            boxes_preds = boxes_preds.reshape(-1, 4)
            boxes_labels = boxes_labels.reshape(-1, 4)
            prefix_size = boxes_preds_shape[:-2]
            # target_size_to_be_reshaped = prefix_size + (M_origin, N_origin)

        expand_size = prefix_size + (M_origin, N_origin, 2)

        """
        Calculates intersection over union
        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
            tensor: Intersection over union for all examples
        """

        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

            boxes_preds_new = torch.zero_(boxes_preds)
            boxes_preds_new[..., 0:1] = box1_x1
            boxes_preds_new[..., 1:2] = box1_y1
            boxes_preds_new[..., 2:3] = box1_x2
            boxes_preds_new[..., 3:4] = box1_y2

            boxes_labels_new = torch.zero_(boxes_labels)
            boxes_labels_new[..., 0:1] = box2_x1
            boxes_labels_new[..., 1:2] = box2_y1
            boxes_labels_new[..., 2:3] = box2_x2
            boxes_labels_new[..., 3:4] = box2_y2

        if box_format == "corners":
            boxes_preds_new = boxes_preds
            boxes_labels_new = boxes_labels

        boxes_preds = boxes_preds_new[:, 2:].reshape(prefix_size + (M_origin, 2))
        boxes_preds_unsqueeze = boxes_preds.unsqueeze(-2)

        boxes_labels = boxes_labels_new[:, 2:].reshape(prefix_size + (N_origin, 2))
        boxes_labels_unsqueeze = boxes_labels.unsqueeze(-3)

        max_xy = torch.min(boxes_preds_unsqueeze.expand(expand_size),
                           boxes_labels_unsqueeze.expand(expand_size))
        min_xy = torch.max(
            boxes_preds_new[:, :2].reshape(prefix_size + (M_origin, 2)).unsqueeze(-2).expand(expand_size),
            boxes_labels_new[:, :2].reshape(prefix_size + (N_origin, 2)).unsqueeze(-3).expand(expand_size))

        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[..., 0] * inter[..., 1]
        # 计算先验框和真实框各自的面积
        area_a = ((boxes_preds_new[:, 2] - boxes_preds_new[:, 0]) * (
                    boxes_preds_new[:, 3] - boxes_preds_new[:, 1])).reshape(prefix_size + (M_origin,)).unsqueeze(
            -1).expand_as(inter)  # [M,N]
        area_b = ((boxes_labels_new[:, 2] - boxes_labels_new[:, 0]) * (
                    boxes_labels_new[:, 3] - boxes_labels_new[:, 1])).reshape(prefix_size + (N_origin,)).unsqueeze(
            -2).expand_as(inter)  # [M,N]
        union = area_a + area_b - inter
        iou_with_adjust = inter / (union + 1e-6)

        return iou_with_adjust


    def intersection_over_union(boxes_preds: torch.tensor, boxes_labels: torch.tensor, box_format="midpoint", eps=1e-6):
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
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
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

            boxes_preds[..., 0:1] = box1_x1
            boxes_preds[..., 1:2] = box1_y1
            boxes_preds[..., 2:3] = box1_x2
            boxes_preds[..., 3:4] = box1_y2

            boxes_labels[..., 0:1] = box2_x1
            boxes_labels[..., 1:2] = box2_y1
            boxes_labels[..., 2:3] = box2_x2
            boxes_labels[..., 3:4] = box2_y2

        if box_format == "corners":
            pass
            #     boxes_preds_new = boxes_preds
            #     boxes_labels_new = boxes_labels
        # boxes_preds = boxes_preds_new[..., 2:].reshape(prefix_size+(M_origin,2))
        boxes_preds_temp = boxes_preds[..., 2:]
        boxes_preds_unsqueeze = boxes_preds_temp.unsqueeze(-2)

        boxes_labels = boxes_labels[..., 2:].reshape(prefix_size + (N_origin, 2))
        boxes_labels_unsqueeze = boxes_labels.unsqueeze(-3)

        max_xy = torch.min(boxes_preds_unsqueeze.expand(expand_size),
                           boxes_labels_unsqueeze.expand(expand_size))
        min_xy = torch.max(boxes_preds[..., :2].reshape(prefix_size + (M_origin, 2)).unsqueeze(-2).expand(expand_size),
                           boxes_labels[..., :2].reshape(prefix_size + (N_origin, 2)).unsqueeze(-3).expand(expand_size))

        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[..., 0] * inter[..., 1]
        # 计算先验框和真实框各自的面积
        area_a = ((boxes_preds[..., 2] - boxes_preds[..., 0]) * (boxes_preds[..., 3] - boxes_preds[..., 1])).reshape(
            prefix_size + (M_origin,)).unsqueeze(-1).expand_as(inter)  # [M,N]
        area_b = ((boxes_labels[..., 2] - boxes_labels[..., 0]) * (
                    boxes_labels[..., 3] - boxes_labels[..., 1])).reshape(prefix_size + (N_origin,)).unsqueeze(
            -2).expand_as(inter)  # [M,N]
        union = area_a + area_b - inter
        iou_with_adjust = inter / (union + 1e-6)

        return iou_with_adjust


    def mean_average_precision(
            pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
    ):
        """
        Calculates mean average precision
        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes
        Returns:
            float: mAP value across all classes given a specific IoU threshold
        """

        # list storing all AP for respective classes
        average_precisions = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)





    def get_bboxes(
            loader,
            model,
            iou_threshold,
            threshold,
            pred_format="cells",
            box_format="midpoint",
            device="cuda",
    ):
        all_pred_boxes = []
        all_true_boxes = []

        # make sure model is in eval before get bboxes
        model.eval()
        train_idx = 0

        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(x)

            batch_size = x.shape[0]
            true_bboxes = cellboxes_to_boxes(labels)
            bboxes = cellboxes_to_boxes(predictions)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    threshold=threshold,
                    box_format=box_format,
                )

                # if batch_idx == 0 and idx == 0:
                #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
                #    print(nms_boxes)

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for box in true_bboxes[idx]:
                    # many will get converted to 0 pred
                    if box[1] > threshold:
                        all_true_boxes.append([train_idx] + box)

                train_idx += 1

        model.train()
        return all_pred_boxes, all_true_boxes


    def convert_cellboxes(predictions, S=7):
        """
        Converts bounding boxes output from Yolo with
        an image split size of S into entire image ratios
        rather than relative to cell ratios. Tried to do this
        vectorized, but this resulted in quite difficult to read
        code... Use as a black box? Or implement a more intuitive,
        using 2 for loops iterating range(S) and convert them one
        by one, resulting in a slower but more readable implementation.
        """

        predictions = predictions.to("cpu")
        batch_size = predictions.shape[0]
        predictions = predictions.reshape(batch_size, 7, 7, 30)
        bboxes1 = predictions[..., 21:25]
        bboxes2 = predictions[..., 26:30]
        scores = torch.cat(
            (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
        )
        best_box = scores.argmax(0).unsqueeze(-1)
        best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
        cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
        x = 1 / S * (best_boxes[..., :1] + cell_indices)
        y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
        w_y = 1 / S * best_boxes[..., 2:4]
        converted_bboxes = torch.cat((x, y, w_y), dim=-1)
        predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
        best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
            -1
        )
        converted_preds = torch.cat(
            (predicted_class, best_confidence, converted_bboxes), dim=-1
        )

        return converted_preds


    def cellboxes_to_boxes(out, S=7):
        converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
        converted_pred[..., 0] = converted_pred[..., 0].long()
        all_bboxes = []

        for ex_idx in range(out.shape[0]):
            bboxes = []

            for bbox_idx in range(S * S):
                bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
            all_bboxes.append(bboxes)

        return all_bboxes


    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)


    def load_checkpoint(checkpoint, model, optimizer):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])


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

        overlap_xmin = max(xA - wA / 2, xB - wB / 2)
        overlap_ymin = max(yA - hA / 2, yB - wB / 2)
        overlap_xmax = min(xA + wA / 2, xB + wB / 2)
        overlap_ymax = min(yA + hA / 2, yB + hB / 2)

        W = overlap_xmax - overlap_xmin
        H = overlap_ymax - overlap_ymin

        if min(W, H) < 0:
            return 0

        intersect = W * H
        union = areaA + areaB - intersect

        return intersect / union


    def xyXY(bboxA, bboxB):
        """
        Similar to the above function, this one computes the intersection over union
        between two 2-d boxes. The difference with this function is that it accepts
        bounding boxes in the form [xmin, ymin, XMAX, YMAX].
        Attributes:
            bboxA (list): defined by 4 values: [xmin, ymin, XMAX, YMAX].
            bboxB (list): defined by 4 values: [xmin, ymin, XMAX, YMAX].
        Returns:
            IOU (float): a value between 0-1 representing how much these boxes overlap.
        """
        xminA, yminA, xmaxA, ymaxA = bboxA
        widthA = xmaxA - xminA
        heightA = ymaxA - yminA
        areaA = widthA * heightA

        xminB, yminB, xmaxB, ymaxB = bboxB
        widthB = xmaxB - xminB
        heightB = ymaxB - yminB
        areaB = widthB * heightB

        xA = max(xminA, xminB)
        yA = max(yminA, yminB)
        xB = min(xmaxA, xmaxB)
        yB = min(ymaxA, ymaxB)

        W = xB - xA
        H = yB - yA

        if min(W, H) < 0:
            return 0

        intersect = W * H
        union = areaA + areaB - intersect

        iou = intersect / union

        return iou


    def box_iou_xywh(box1, box2):
        x1min, y1min = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
        x1max, y1max = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
        s1 = box1[2] * box1[3]

        x2min, y2min = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
        x2max, y2max = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
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
        a = torch.tensor([100, 50, 200, 100])
        b = torch.tensor([100, 100, 200, 200])
        c = torch.tensor([110., 100., 200., 200.])

        a2 = torch.tensor([80, 50, 200, 100])
        b2 = torch.tensor([50, 100, 200, 200])
        c2 = torch.tensor([120., 120., 220., 220.])

        A = [a, b, c]
        B = [a2, b2, c2]
        for i in range(3):
            for j in range(3):
                print("1 xywh:", xywh(A[i], B[j]))
                print("2 box_iou_xywh:", box_iou_xywh(A[i], B[j]))
                print("3 xywh calculates_ious", calculates_ious(A[i].unsqueeze(0), B[j].unsqueeze(0)))

                print("4 bbox_iou(A[i], B[j])",bbox_iou(A[i].unsqueeze(0), B[j].unsqueeze(0)))

                print("5 corners calculates_ious", calculates_ious(A[i].unsqueeze(0), B[j].unsqueeze(0),box_format = "corners"))



        print(calculates_ious(torch.stack((a, b, c), dim=0), torch.stack((a2, b2, c2), dim=0), eps=0))
        print(calculates_ious(torch.stack((a, b), dim=0), torch.stack((a2, b2, c2), dim=0), eps=0))

        print(calculates_ious(torch.stack((a, b, c), dim=0), torch.stack((a2, b2), dim=0), eps=0))
        At1 = torch.stack((a, b, c))
        At2 = torch.stack((a2, b2, c2))
        print(calculates_ious(torch.stack((At1, At2), dim=0), torch.stack((At2, At1), dim=0), eps=0))












