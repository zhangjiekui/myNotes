import numpy as np  # 可能用到的数据值计算库
import os           # 可能用到的文件操作
import matplotlib.pyplot as plt   		# 图形绘制 
import matplotlib.patches as patches 	# 添加矩形框
import matplotlib.image as image  		# 读取图像数据

def BoundingBox_Denote(bbox=[], mode=True): # xyxy or xywh
    '''边界框的表示形式的转换
        bbox: 包含(x1, y1, x2, y2)四个位置信息的数据格式
        mode: 边界框数据表示的模式
             True:  to (x1,y1,x2,y2) #左上顶点 (x1,y1) ，右下顶点 (x2,y2 ) 
             False: to (x,y,w,h)     #  中心点 (x0,y0)， w是边界框的宽， h是边界框的长(高)
        return: 返回形式转换后的边界框数据
    '''
    denote_bbox = [] # 转换表示的边界框

    if mode is True:  # 保持原形式
        denote_bbox = bbox
    else:  # 转换为(center_x, center_y, w, h)
        center_x = (bbox[0]+bbox[2]) / 2.0
        center_y = (bbox[1]+bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        denote_bbox = [center_x, center_y, w, h]
    
    # 返回表示转换的边界框表示
    denote_bbox = np.asarray(denote_bbox,  dtype='float32')
    return denote_bbox

def draw_rectangle(bbox=[], mode=True,linewidth=1, color='k', fill=True):
    '''绘制矩形框
        bbox：边界框数据（默认框数据不超过图片边界）
        mode: 边界框数据表示的模式
             True:  to (x1,y1,x2,y2)
             False: to (x,y,w,h)
        color: 边框颜色
        fill: 是否填充
    '''
    if mode is True: # to (x1,y1,x2,y2)
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1
    else: # to (x,y,w,h)
    	# 默认绘制的框不超出边界
        x = bbox[0] - bbox[2] / 2.0
        y = bbox[1] - bbox[3] / 2.0
        w = bbox[2]
        h = bbox[3]
    
    # 绘制边界框
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    # 获取绘制好的图形的返回句柄——用于添加到当前的图像窗口中
    rect = patches.Rectangle((x, y), w, h, 
                             linewidth=linewidth,        # 线条宽度
                             edgecolor=color,    # 线条颜色
                             facecolor='y',      # 
                             fill=fill, 
                             linestyle='-')
    
    return rect

def img_draw_bbox(img,bbox=[10, 20, 90, 100], mode=True,offset=15,if_draw_text=False):
    '''将边界框绘制到实际图片上
        bbox: 需要绘制的边界框
        mode: 边界框数据表示的转换模式
             True:  to (x1,y1,x2,y2)
             False: to (x,y,w,h)
    '''
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()  # 窗口句柄

    # 图片路径
    if isinstance(img,str):
        # img_path = os.path.join(os.getcwd(), 'img', '1.jpg')
        img = image.imread(img) # 读取图片数据
    plt.imshow(img)  # 展示图片

    # 边界框数据转换
    denote_mode = mode  # 边界框表示形式——确定数据格式
    # 经过转换后的边界框数据
    bbox1 = BoundingBox_Denote(bbox=bbox, mode=denote_mode)

    # 绘制表示模式2的边界框
    rect1 = draw_rectangle(bbox=bbox1, mode=denote_mode, color='r')
    ax.add_patch(rect1)  # 将矩形添加到当前的图片上
    if if_draw_text:
        plt.text(bbox[0]+offset,bbox[1]+offset,s=f"({bbox[0]},{bbox[1]})", color='r')
        plt.text(bbox[2]-5*offset,bbox[3]-offset/2,s=f"({bbox[2]},{bbox[3]})", color='r')

    plt.show()

def draw_anchor(ax, center, length, scales, ratios, img_height, img_width, colors=['r','b'],offset=15,if_draw_text=False):
    '''绘制锚框————同一中心点三个不同大小的锚框
        ax: plt的窗体句柄——用于调用矩形绘制
        center：中心点坐标
        length：基本长度
        scales：尺寸
        ratios：长宽比
        img_height: 图片高
        img_width: 图片宽

        一个锚框的大小，由基本长度+尺寸+长宽比有关
        同时锚框的最终计算值与图片实际大小有关——不能超过图片实际范围嘛
    '''

    bboxs = []  # 这里的边界框bbox是指的锚框

    for scale in scales: # 遍历尺寸情况
        
        for ratio in ratios: # 同一尺寸下遍历不同的长宽比情况
            print(f"    \n-------------------------scale: {scale} of {scales},ratio: {ratio} of {ratios}------------------------------")
            # 利用基本长度、尺寸与长宽比进行锚框长宽的转换
            h = length * scale * np.math.sqrt(ratio)
            w = length * scale / np.math.sqrt(ratio)
            # 利用求得的长宽，确定绘制矩形需要的左上角顶点坐标和右下角顶点坐标
            # 不同的绘制API可能有不同的参数需要，相应转换即可
            x1 = max(center[0] - w / 2., 0.)  # 考虑边界问题
            y1 = max(center[1] - h / 2., 0.)
            x2 = min(center[0] + w / 2. - 1.0, img_width - 1.)  # center[0] + w / 2 -1.0 是考虑到边框不超过边界
            y2 = min(center[1] + h / 2. - 1.0, img_height - 1.)
            
            bbox = [x1, y1, x2, y2]
            # print(f'    An Anchor({center[0]},{center[1]},w={int(w),int(x2-x1)},h={int(h),int(y2-y1)}): ', bbox)
            print(f'         An Anchor({center[0]},{center[1]},w={int(w)},h={int(h)}): ', bbox)
            bboxs.append(bbox)  # 插入生成的anchor

    _index_of_s = 0
    for bbox in bboxs:
        # print(_index_of_s)
        if _index_of_s<len(ratios):
            color=colors[0]
        else:
            color=colors[1]
        _index_of_s+=1
        denote_mode = True  # 当前的目标数据形式： True: (x1, y1, x2, y2)
        denote_bbox = BoundingBox_Denote(bbox=bbox, mode=denote_mode)

        # 绘制anchor的矩形框
        rect = draw_rectangle(bbox=denote_bbox, mode=denote_mode, color=color,fill=False)
        if if_draw_text:

            plt.text(bbox[0]+offset,bbox[1]+offset,s=f"({bbox[0]},{bbox[1]})", color='r')
            plt.text(bbox[2]-5*offset,bbox[3]-offset/2,s=f"({bbox[2]},{bbox[3]})", color='r')
        
        ax.add_patch(rect)

def draw_anchors(img):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 图片路径
    if isinstance(img,str):
        img = image.imread(img) # 读取图片数据
    plt.imshow(img)  # 展示图片
    print(f"原始图片的尺寸为（{img.shape[0]},{img.shape[1]}）")

    num=0
    for i in range(100, img.shape[0], 200):  # y值
        for j in range(90, img.shape[1], 200): # x值
            num+=1

            # center: x, y            
            y = i 
            x = j
            print(f"  num_centers{num}:center({i,j})")
            draw_anchor(ax=ax, center=[x, y], 
                        length=120, scales=[0.5,1.0], ratios=[0.5,1.0, 1.5], 
                        img_height=img.shape[0], img_width=img.shape[1],
                        colors=['r','b'])
    
    plt.show()


if __name__ == '__main__':
    # 边界框真实数据
    test_bbox = [160, 60, 350, 260]

    # 边界框数据表示模式——输入的bbox数据必须是[x1,y1,x2,y2]
    # True:  to (x1,y1,x2,y2)
    # False: to (x,y,w,h)
    # denote_mode = True
    denote_mode = False

    # 测试边界框的转换是否成功
    _test_bbox = BoundingBox_Denote(bbox=test_bbox, mode=denote_mode)
    # print(denote_mode,_test_bbox)

    # 测试边界框的绘制
    img_draw_bbox(img="/mnt/d/dockerv/megengine_practice/imgs/1.jpg",bbox=test_bbox, mode=denote_mode,if_draw_text=True)
    print("=====")
    print("====="*10)    
    print("=====")
    draw_anchors(img="/mnt/d/dockerv/megengine_practice/imgs/1.jpg")



