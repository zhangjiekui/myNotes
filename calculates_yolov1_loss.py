class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        assert (targets < 0).sum() == 0
        prediction_boxes= predictions[..., 20:]
        prediction_boxes = prediction_boxes.reshape(-1, self.S, self.S, self.B , 5)
        prediction_boxes = prediction_boxes[..., 0:4]
        assert (targets < 0).sum() == 0
        target_boxes= targets[..., 20:]
        target_boxes = target_boxes.reshape(-1, self.S, self.S, self.B , 5)
        target_boxes = target_boxes[..., 0:4]

        test_ious = calculates_ious(prediction_boxes[0, 3, 2], target_boxes[0, 3, 2])
        print(test_ious)
        iou_maxes, bestbox = torch.max(test_ious, dim=-1)
        iou_maxes_index = iou_maxes>0
        bestbox_index = bestbox * iou_maxes_index
        assert (targets < 0).sum() == 0
        # Calculate IoU for the two predicted bounding boxes with target bbox
        ious = calculates_ious(prediction_boxes, target_boxes)
        assert (targets < 0).sum() == 0
        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        # ious[0,3,2,0,0]=1
        # ious[0,3,2, 0, 1] = 2
        # ious[0,3,2,1,0]=3
        # ious[0,3,2, 1, 1] = 4
        iou_maxes, bestbox = torch.max(ious, dim=-1)
        # ious[0, 0, 0, 1, 0] = 8
        bs,S,S,Bpre,Btgt = ious.size()
        ious = ious.reshape(bs,S,S,-1)
        iou_maxes, bestbox = torch.max(ious, dim=-1)
        iou_maxes_index = iou_maxes>0
        bestbox_index = bestbox * iou_maxes_index

        # bestbox_index里的取值为[0,1,2,3],即[IOU_00,IOU_01,IOU_10,IOU_11]的序号
        # bestbox_index < 2 则取第0个Prediction Bbox
        # bestbox_index = 0 或 2 则取第0个Target Bbox
        assert (targets < 0).sum() == 0
        bestbox_predict_index = bestbox_index < 2
        bestbox_predict_index = bestbox_predict_index.int().unsqueeze(3)

        bestbox_target_index = (bestbox_index == 0) + (bestbox_index == 2)
        bestbox_target_index = bestbox_target_index.int().unsqueeze(3)

        exists_box = targets[..., 24].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox_predict_index * predictions[...,20:24] ##如果序号为1或者True，表明是第1个bbox最好，所以索引为【20:24】
                + (1 - bestbox_predict_index) * predictions[..., 25:29] #如果序号为0或者False，表明是第2个bbox最好，所以索引为【25:29】
            )
        )
        assert (targets < 0).sum() == 0
        # box_targets = exists_box * target[..., 21:25]
        box_targets = exists_box * (
            (
                    bestbox_target_index * targets[..., 20:24]  #如果序号为1或者True，表明是第1个bbox最好，所以索引为【20:24】
                    + (1 - bestbox_target_index) * targets[..., 25:29] #如果序号为0或者False，表明是第2个bbox最好，所以索引为【25:29】
            )
        )
        assert (targets < 0).sum() == 0

        # Take sqrt of width, height of boxes to ensure that 并且保证正负号存在（torch.sign）
        # 处理完torch.sqrt（宽高）后，放入box_predictions
        # 最后一次性求mse(均方损失函数)
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        assert (targets < 0).sum() == 0
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox_predict_index * predictions[..., 24:25] + (1 - bestbox_predict_index) * predictions[..., 29:30]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., 24:25]),
            # target[..., 29:30] 可以不用
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #


        # 第一个pred_box的no_object_loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 24:25], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 24:25], start_dim=1),
        )
        # 加上第二个pred_box的no_object_loss
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 29:30], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 24:25], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        target_class = targets[..., :20]
        target_class = (target_class>=1).int() # 因为在数据处理时，如果一个Cell中，第二个target存在，则对应的class_id设为了2，所以这里要进行对应的处理
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target_class, end_dim=-2,),
        )
        
        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )
        print(f"中心点坐标和框宽高sqrt的box_loss={box_loss}，object_loss={object_loss}，no_object_loss={no_object_loss}，class_loss={class_loss}，加权汇总后的总误差={loss}")
        return loss
