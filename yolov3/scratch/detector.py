from __future__ import division
import time
import cv2
from util import *
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
from model_configs.classes_names_for_dataset import COCO_CLASSES
from util import plt_plot_bbox_on_image
from darknet import non_max_suppression
# import matplotlib.colors as mcolors#
# COLORS_LIST = list(mcolors.CSS4_COLORS.values())
COLORS_LIST = pkl.load(open("pallete", "rb"))

def letterbox_image(original_img, img_dims_to_be_resized):
    """
    resize image with unchanged aspect ratio using padding
    letterbox_image resizes the image, but keeping the aspect ratio consistent,
    and padding the left out areas with the color (128,128,128)
    """
    img_w, img_h = original_img.shape[1], original_img.shape[0]
    w, h = img_dims_to_be_resized
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((img_dims_to_be_resized[1], img_dims_to_be_resized[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas

def prep_image(original_img, img_dims_to_be_resized):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """
    original_img = letterbox_image(original_img, (img_dims_to_be_resized, img_dims_to_be_resized))
    #  PyTorch's image input format is (Batches x Channels x Height x Width),
    #  with the channel order being RGB
    original_img = original_img[:, :, ::-1].transpose((2, 0, 1)).copy() # img[:, :, ::-1]的作用 BGR-->RGB
    original_img = torch.from_numpy(original_img).float().div(255.0).unsqueeze(0)
    return original_img

def write(bbox:torch.Tensor, original_imgs_array_list: list,colors = COLORS_LIST):
    # print(bbox)
    c1 = tuple(bbox[0:2].int().numpy())
    c2 = tuple(bbox[2:4].int().numpy())
    img = original_imgs_array_list[int(bbox[-1])]
    cls = int(bbox[-2])
    # color = COLORS_LIST[cls]
    color = colors[cls]
    label = "{0}".format(COCO_CLASSES[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

def letterbox_revert_back(im_original_dim_list:list, predictions_tensor:torch.Tensor,im_resized_dim_for_net_input=608):
    output = predictions_tensor.cpu()
    inp_dim = im_resized_dim_for_net_input
    im_original_dim_list = torch.index_select(im_original_dim_list, dim=0, index=output[:, -1].long())
    scaling_factor = torch.min(inp_dim / im_original_dim_list, dim=1)[0].view(-1, 1)
    output[:, [0, 2]] -= (inp_dim - scaling_factor * im_original_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_original_dim_list[:, 1].view(-1, 1)) / 2
    output[:, 0:4] /= scaling_factor
    for i in range(output.shape[0]):
        # torch.clamp(input,min,max)
        output[i, [0, 2]] = torch.clamp(output[i, [0, 2]], 1.0, im_original_dim_list[i, 0]-1.0)
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 1.0, im_original_dim_list[i, 1]-1.0)
    return output

def cv2_draw_bbox_on_image(imgs:list,bboxes:torch.Tensor,im_path_list:list,destination:str):
    """
    imgs:list  带绘制的img列表
    bboxes:torch.Tensor 待绘制的bboxes
    """
    list(map(lambda bbox: write(bbox, imgs), bboxes))
    # list(map(lambda x: write(x,bboxes), imgs))
    det_names = pd.Series(im_path_list).apply(lambda x: "{}{}det_{}".format(destination, os.sep, x.split(os.sep)[-1]))
    # return list(map(cv2.imwrite, det_names, _loaded_ims))
    return list(map(cv2.imwrite, det_names, list(imgs.values())))

# for cfg in [yolov3_tiny]:
# for cfg in [yolov3]:
CUDA = torch.cuda.is_available()
# classes = COCO_CLASSES
num_classes = len(COCO_CLASSES)
img_path = r'D:\dockerv\yolov3\dog-cycle-car.png'
yolov3=["yolov3",r"D:\dockerv\yolov3\scratch\model_configs\yolov3.cfg"]               # layer106_yolo.shape=torch.Size([1, 17328, 85])
yolov3_tiny=["yolov3-tiny",r"D:\dockerv\yolov3\scratch\model_configs\yolov3-tiny.cfg"]     # layer23_yolo.shape=torch.Size([1, 2028, 85])
# Spatial Pyramid Pooling（空间金字塔池化结构）
yolov3_spp=["yolov3-spp",r"D:\dockerv\yolov3\scratch\model_configs\yolov3-spp.cfg"]       # layer113_yolo.shape=torch.Size([1, 17328, 85])

# for cfg in [yolov3_spp]:
for cfg in [yolov3_tiny,yolov3,yolov3_spp]:
    logger.error(f"==={cfg[0]}"*10)
    model = Darknet(cfgfile=cfg[1])
    weightfile=f"D:\dockerv\yolov3\scratch\model_configs\{cfg[0]}.weights"
    model.load_weights(weightfile)
    width = int(model.net_info['width'])
    height = int(model.net_info['height'])
    assert width==height,f'Yolo系列中，网络输入图片的宽高应该相等,但现在宽={width},高={height} !'
    assert width % 32 == 0 ,f'Yolo系列中，网络输入图片的宽高（{width}）应该能被32整除 !'
    assert width > 32 ,f'Yolo系列中，网络输入图片的宽高（{width}）应该大于32 !'

    model.eval()
    images = 'imgs'
    destination_folder = 'imgs_detect_results'
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    destination = osp.join(osp.realpath('.'), destination_folder)
    if not os.path.exists(destination):
        os.makedirs(destination)

    loaded_ims = [cv2.imread(x) for x in imlist]
    #List containing dimensions of original images
    im_original_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    # 当参数只有两个时：（列的重复倍数，行的重复倍数）
    # 当参数有三个时： （通道数的重复倍数，列的重复倍数，行的重复倍数）
    # 参数列表的理解：对应tensor维度的重复倍数。
    # The number of times to repeat this tensor along each dimension
    # im_original_dim_list = torch.FloatTensor(im_original_dim_list).repeat(1,2)
    # 实际上repeat并没有需要
    im_original_dim_list = torch.FloatTensor(im_original_dim_list)
    imgs_dims_to_be_resized = [width]*len(imlist)

    im_batches_list = list(map(prep_image, loaded_ims, imgs_dims_to_be_resized))

    if CUDA:
        model.cuda()
        im_original_dim_list.cuda()

    batch_size = 6
    leftover = 0
    if (len(imlist) % batch_size):
        leftover = 1
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches_list[i * batch_size:
                                        min((i + 1) * batch_size,len(im_batches_list))])) for i in range(num_batches)]
    start_det_loop = time.time()
    predictions = []
    for i, batch in enumerate(im_batches):
        #load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        # (cX,cY,w,h)
        prediction = model(batch, CUDA) #[bs,   total_num_anchors,   5(cX,cY,w,h,confidence) + 80(num_classes)]
        # [Tensor(num_boxes, 8) or None]，
        # 其中8位的含义是4个坐标(x,y,x,y),1个obj_conf,1个cls_conf,1个类别id,1个idx_batch
        # (x,y,x,y)
        prediction = non_max_suppression(prediction,conf_thres=0.5,nms_thres=0.4,use_soft_nms=False)
        end = time.time()
        elapsed = end - start
        logger.info("Batch{0:2d} 总耗时 {1:6.3f}Seconds，平均耗时 {2:6.3f}Seconds/Image".format(i, elapsed ,elapsed / batch_size))

        if prediction is None:
            logger.warning(f"    Batch {i} have detected nothing!")
            continue
        else:
            prediction[:, -1] += i * batch_size  # transform the atribute from index in batch to index in imlist
            predictions.append(prediction)
        # if CUDA:
        #     torch.cuda.synchronize()
            # makes sure that CUDA kernel is synchronized with the CPU.
            # Otherwise, CUDA kernel returns the control to CPU as soon as the GPU job
            # is queued and well before the GPU job is completed (Asynchronous calling).
            # This might lead to a misleading time if end = time.time() gets printed before the GPU job is actually over.

    predictions_tensor = torch.cat(predictions)
    image_idx_set_has_detected_bboxes = set(predictions_tensor[...,-1].tolist())
    image_idx_set_has_detected_bboxes = set(int(i) for i in image_idx_set_has_detected_bboxes)
    total_image_idx = set(range(len(imlist)))
    has_no_bbox = total_image_idx - image_idx_set_has_detected_bboxes
    for img_idx in has_no_bbox:
        logger.warning("    Image{0:2d} [{1:20s}] has detected nothing!".format(img_idx, imlist[img_idx]))



    # 将letterbox变换后的坐标predictions_tensor，缩放回im_original_dim_list的大小
    predictions_tensor = letterbox_revert_back(im_original_dim_list, predictions_tensor, im_resized_dim_for_net_input=width)
    # cv2画图和Bbox

    # imgs_have_bboxes = [loaded_ims[idx] for idx in image_idx_set_has_detected_bboxes]
    imgs_have_bboxes = {idx:loaded_ims[idx] for idx in image_idx_set_has_detected_bboxes}
    imgs_path_list = [imlist[idx] for idx in image_idx_set_has_detected_bboxes]
    # if_draw_succeeded = cv2_draw_bbox_on_image(loaded_ims,predictions_tensor)
    if_draw_succeeded = cv2_draw_bbox_on_image(imgs_have_bboxes,predictions_tensor,im_path_list=imgs_path_list,destination=destination)

    result_print = list(zip(imgs_path_list,if_draw_succeeded))
    for r in result_print:
        if r[1]:
            logger.info(r)
        else:
            logger.warning(r)
    # plt画图和Bbox
    for img_idx in image_idx_set_has_detected_bboxes:
        mask = predictions_tensor[:,-1] == img_idx
        detections_on_images = predictions_tensor[mask]
        assert len(detections_on_images)>0, "这里只保留了检测出有目标的列表，所以长度应该大于0！"
        plt_plot_bbox_on_image(imlist[img_idx], bbox=detections_on_images, mode="xyxy", bbox_has_scaled=True, if_draw_text=True, fill=True,
                                   wh_net_output=(width, height))




torch.cuda.empty_cache()
