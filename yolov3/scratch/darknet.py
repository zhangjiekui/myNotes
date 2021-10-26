import os
import cv2
import matplotlib.image as image  # 读取图像数据
import torch
import torch.nn as nn
import torch.nn.functional as F 
# from torch.autograd import Variable
import numpy as np
from util_mylogger import logger
from util import predict_transform,write_results,non_max_suppression,plot_image

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_yolov3_cfg(cfgfile='scratch\yolov3.cfg'):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    cfgfile = os.path.abspath(cfgfile)
    logger.info(f'yolov3.cfg file path={cfgfile}')

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.strip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    for line in lines:
        # This marks the start of a new block
        if line[0] == '[':  
            # If block is not empty, implies it is storing values of previous block.                       
            if len(block) != 0:
                blocks.append(block)  # add it the to blocks list
                block = {}            # re-init the block
            block['type'] = line[1:-1].strip()
        else:
            key,value = line.split("=")    
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3 # 持续跟踪应用卷积的层的filters数量,初始化为3，因为图像具有对应于RGB通道的3个filters
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]

            # 有些convolutional中没有batch_normalize参数，分别是[81,93,105]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"):            
            route = EmptyLayer()            
            module.add_module("route_{0}".format(index), route)
            # FIXME:为了兼容yolov3_spp，其route layers其实有4个
            # 可能是此处的filter没变过来
            x["layers"] = x["layers"].split(',')
            idx_layers = [int(x) for x in x["layers"]]
            transformed_idx_layers = [x if x>0 else index+x for x in idx_layers] 
            
            num_routes = len(transformed_idx_layers)
            filters = 0
            for i in range(num_routes):
                idx = transformed_idx_layers[i]
                filters = filters + output_filters[idx]

            

            # if end < 0:
            #     filters = output_filters[index + start] + output_filters[index + end]
            # else:
            #     filters= output_filters[index + start]
            # FIXME:可能是此处的filter没变过来
            
            # idx_layers = [int(x) for x in x["layers"]]
            # if len(idx_layers)>1:
            #     print(idx_layers)
            # transformed_idx_layers = [x if x>0 else index+x for x in idx_layers ]          
            # #Start  of a route
            
            # start = int(x["layers"][0])
            # if len(x["layers"])>1:
            #     print(f'layers={len(x["layers"])},x["layers"]')
            # #end, if there exists one.
            # try:
            #     end = int(x["layers"][1])
            # except:
            #     end = 0
            # #Positive anotation
            # if start > 0: 
            #     start = start - index
            # if end > 0:
            #     end = end - index
            # route = EmptyLayer()
            # module.add_module("route_{0}".format(index), route)
            # if end < 0:
            #     filters = output_filters[index + start] + output_filters[index + end]
            # else:
            #     filters= output_filters[index + start]
            
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        # TODO:重构了，为了兼容yolov3_tiny.cfg中的maxpool
        elif x["type"] == "maxpool":
            size = int(x["size"])
            stride = int(x["stride"])
            padding=(size-1)//2
            if stride == 1 and size==2: 
                #             nn.ZeroPad2d(num左，num右，num上，num下）
                zeropadding = nn.ZeroPad2d((0, 1, 0, 1)) # TODO:为了保持in=13*13，out也等于13*13
                module.add_module("zeropadding_{}".format(index), zeropadding)
            maxpool = nn.MaxPool2d(kernel_size=size,stride=stride,padding=padding)
            module.add_module("maxpool_{}".format(index), maxpool)

                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    logger.info(module_list)    
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile='scratch\model_configs\yolov3.cfg'):
        super(Darknet, self).__init__()
        self.blocks = parse_yolov3_cfg(cfgfile)
        self.modules_corresponding_to_blocks = self.blocks[1:]
        self.net_info, self.module_list = create_modules(self.blocks)
        assert len(self.module_list)==len(self.modules_corresponding_to_blocks)

    def forward(self,x,CUDA=torch.cuda.is_available()):
        outputs = {}   #We cache the outputs for the route layer
        
        detections_list=[]
        yolo_layer_idx=[]
        #TODO: 重构了
        # write = 0 


        for i,block in enumerate(self.modules_corresponding_to_blocks):        
            module_type = (block["type"])
            # TODO: 重构了，为了兼容yolov3_tiny.cfg中的maxpool
            if module_type in ["convolutional","upsample","maxpool"]:
                # if i==11 and module_type=="maxpool": #debug tiny
                # if i==82 and module_type=="maxpool":   #debug spp
                # if i==83 :                
                #     print('pause')
                #     print(self.modules[i])
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = block["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                # 如果route中只有一个layer_idx，则直接获取对应layer_idx的输出
                if len(layers) == 1:
                    x = outputs[i + (layers[0])] #则直接获取对应layer_idx的输出            
                
                # 如果route中不只一个layer_idx，则获取所有对应layer_idx的输出（多个）,并且cat在一起
                else:
                    maps = []
                    for j in range(len(layers)): #获取所有对应layer_idx的输出（多个）
                        if (layers[j]) > 0:
                            layers[j] = layers[j] - i
                        map = outputs[i + layers[j]]
                        maps.append(map)
                    # x = torch.cat((map1, map2), 1)
                    x = torch.cat(maps,1) # cat在一起

                # else:
                #     if (layers[1]) > 0:
                #         layers[1] = layers[1] - i
    
                #     map1 = outputs[i + layers[0]]
                #     map2 = outputs[i + layers[1]]
                #     x = torch.cat((map1, map2), 1)
     
            elif  module_type == "shortcut":
                from_ = int(block["from"])
                x = outputs[i-1] + outputs[i+from_]

            # YOLO (Detection Layer)
            elif module_type == 'yolo': 
                yolo_layer_idx.append(i)       
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (block["classes"])
        
                #Transform 
                x = x.data #shape:bs,depth,h,w
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                detections_list.append(x)
                logger.warning(f"     -----yolo_module DetectionLayer({i}) output [bs,total_num_anchors,5(cX,cY,w,h,confidence) + {num_classes}(num_classes)]={x.shape}\n")

                #TODO: 重构了
                # if not write:              #if no collector has been intialised. 
                #     detections = x
                #     write = 1
                #     logger.warning(f"     -----yolo_module(DetectionLayer{i}),cat之前{detections.shape}\n")
        
                # else:
                           
                #     detections = torch.cat((detections, x), 1)
                #     logger.warning(f"     -----yolo_module(DetectionLayer{i}),cat之后{detections.shape}\n") 
                #TODO: 重构了
                # FIXME:

            # logger.error(f'layer{i}_{module_type}.shape={x.shape}')
            outputs[i] = x
        
        detections_from_list=torch.cat(detections_list,1)
        logger.warning(f"     -----全部yolo_module DetectionLayer({yolo_layer_idx}) 的输出Cat之后的shape={detections_from_list.shape}")
        logger.warning(f"     -----含义为[bs,   total_num_anchors,   5(cX,cY,w,h,confidence) + 80(num_classes)]\n")
        
        return detections_from_list
    
    def load_weights(self, weightfile=r"D:\dockerv\yolov3\scratch\model_configs\yolov3-tiny.weights"):
        #Open the weights file
        logger.warning(f"......载入权重参数:{weightfile}")
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        num_parameters_of_torch_model = sum(x.numel() for x in self.module_list.parameters())
        num_parameters_of_pretrain_model =  torch.from_numpy(weights).numel()

        num_parameters_of_bn_state_dict=0
        num_bn_layers = 0
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    num_bn_layers+=1
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    num_parameters_of_bn_state_dict+= num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    num_parameters_of_bn_state_dict+= num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
        logger.warning(f"num_parameters_of_torch_model = {num_parameters_of_torch_model}")
        logger.warning(f"num_parameters_of_pretrain_model = {num_parameters_of_pretrain_model}")
        logger.warning(f"-------------weight_file_end_idx = {ptr}")
        logger.warning(f"num_diff_(torch_model-pretrain_model) = {num_parameters_of_torch_model-num_parameters_of_pretrain_model}")
        logger.warning(f"----num_bn_layers={num_bn_layers},num_bn_state_dict = {num_parameters_of_bn_state_dict}")    
        
        assert num_parameters_of_torch_model+num_parameters_of_bn_state_dict==num_parameters_of_pretrain_model,"载入的模型参数数量不匹配！"
        logger.warning(f"......成功载入权重参数!") 
    
def get_test_input(img_path=r"D:\dockerv\yolov3\dog-cycle-car.png",resized_shape=(608,608)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, resized_shape)          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W X C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    # img_ = torch.autograd.Variable(img_)                     # Convert to Variable
    img_.requires_grad=True
    return img_


if __name__ == '__main__':
    # blocks=parse_yolov3_cfg()
    # net_info, module_list = create_modules(blocks)

    # print(net_info)
    # print(module_list)
    # logger.warning(module_list)

    # for index, block in enumerate(blocks[1:]):
    #     if block['type']=='convolutional':
    #         print(index,block.keys())

    # from _util_plot import img_draw_bbox
    from util import plt_plot_bbox_on_image
    img_path = r'D:\dockerv\yolov3\dog-cycle-car.png'


    yolov3=["yolov3",r"D:\dockerv\yolov3\scratch\model_configs\yolov3.cfg"]               # layer106_yolo.shape=torch.Size([1, 17328, 85])
    yolov3_tiny=["yolov3-tiny",r"D:\dockerv\yolov3\scratch\model_configs\yolov3-tiny.cfg"]     # layer23_yolo.shape=torch.Size([1, 2028, 85])
    # Spatial Pyramid Pooling（空间金字塔池化结构）
    yolov3_spp=["yolov3-spp",r"D:\dockerv\yolov3\scratch\model_configs\yolov3-spp.cfg"]       # layer113_yolo.shape=torch.Size([1, 17328, 85])

    for cfg in [yolov3_tiny,yolov3,yolov3_spp]:
    # for cfg in [yolov3_spp]:
        logger.error(f"==={cfg[0]}"*10)
        net = Darknet(cfgfile=cfg[1])

        weightfile=f"D:\dockerv\yolov3\scratch\model_configs\{cfg[0]}.weights"
        # weightfile = weightfile.replace('.cfg.','.')
        net.load_weights(weightfile)
        width = int(net.net_info['width'])
        height = int(net.net_info['height'])

        x = get_test_input(img_path=img_path,resized_shape=(width, height))


        # x = torch.randn(2,3,width,height)
        logger.warning(f"\n\n原始输入的形状：{x.shape}")
        out = net(x)
        import copy
        _out = copy.deepcopy(out)
        # r = write_results(out)
        nms_thres=[0,0.01, 0.1 ,0.4,0.9,1.0]
        conf_thres = 0.4
        soft_conf_thres = 0.4
        nms_thres = [0.4]

        r=None
        s=None

        for _thres in nms_thres:
            r = non_max_suppression(out, conf_thres=conf_thres,nms_thres=_thres,use_soft_nms=False)
            logger.error(f'non_max_suppression_nms:_thres={_thres}')
            r_soft = non_max_suppression(out, conf_thres=conf_thres, nms_thres=_thres, use_soft_nms=True,
                                         soft_conf_thres=soft_conf_thres)
            logger.error(f'non_max_suppression_nms_soft:_thres={_thres}')
            logger.error(f'########################################################')
            print(r.shape,r_soft.shape)
            assert r.shape==r_soft.shape

            img = x.detach().numpy()[0]
            img = np.transpose(img, (1, 2, 0))
            plt_plot_bbox_on_image(img, bbox=r_soft[:, 0:4], mode="xyxy", if_draw_text=True, fill=False,
                                   wh_net_output=(width, height))

            plt_plot_bbox_on_image(img_path, bbox=r_soft, mode="xyxy", if_draw_text=True, fill=False,
                                   wh_net_output=(width, height))

            plt_plot_bbox_on_image(img, bbox=r[:, 0:4], mode="xyxy", if_draw_text=True, fill=False, wh_net_output=(width, height))

            plt_plot_bbox_on_image(img_path, bbox=r, mode="xyxy", if_draw_text=True, fill=False,
                                   wh_net_output=(width, height))

            bboxes = [99.1971, 241.6673, 253.0763, 564.6666]
            plt_plot_bbox_on_image(img_path, bbox=bboxes, mode="xyxy", if_draw_text=False, fill=False, wh_net_output=(width, height))
            bboxes = [[ 99.1971, 241.6673, 253.0763, 564.6666],
            [ 84.2191, 123.8694, 469.2323, 475.5970],
            [373.5568,  83.2922, 544.0130, 181.9760]]
            plt_plot_bbox_on_image(img_path, bbox=bboxes, mode="xyxy", if_draw_text=False, fill=False,
                                   wh_net_output=(width, height))



        # plot_image(img,r[:,0:4])
        # plot_image(img, r_soft[:, 0:4])
        # plot_image(img, s[:, 0:4])

        # xyxy
        # print("原始xyxy：",r_soft)
        # img_draw_bbox(img_path, bbox=r_soft[:, 0:4], mode="xyxy", if_draw_text=True, fill=False, wh_net_output=(width, height))
        # img_draw_bbox(img, bbox=r_soft[:, 0:4], mode="xyxy",if_draw_text=True,fill=False,wh_net_output=(width,height))
        # bboxes = [[ 99.1971, 241.6673, 253.0763, 564.6666],
        # [ 84.2191, 123.8694, 469.2323, 475.5970],
        # [373.5568,  83.2922, 544.0130, 181.9760]]
        # bboxes = [ 99.1971, 241.6673, 253.0763, 564.6666]
        # img_draw_bbox(img, bbox=bboxes, mode="xyxy", if_draw_text=True, fill=False,wh_net_output=(width, height))
        # img_draw_bbox(img_path, bbox=bboxes, mode="xyxy", if_draw_text=True, fill=False,
        #               wh_net_output=(width, height))
        print("end")



    # 测试边界框的绘制



    # r = non_max_suppression(out,nms_thres = 0.4)
        # logger.info(f'write_results finished,r.shape={r.shape}')


        # print(cfg,out.shape)
        

        # https://www.cnblogs.com/pprp/p/12432562.html
        # https://zhuanlan.zhihu.com/p/36998818 从零开始实现YOLO v3（part4）https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-4/

        
        # 从零开始学习YOLOv3
        # https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4MjY4NTk0NQ==&action=getalbum&album_id=1344182761996877824&scene=173&from_msgid=2247484666&from_itemidx=2&count=3&nolastread=1#wechat_redirect
