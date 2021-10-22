import os,sys
from  loguru import logger

import os


class ProjectUtil:
    @staticmethod
    def project_root_path(project_name="yolov3", print_log=False):
        """
        获取当前项目根路径
        :param project_name: 项目名称
                                1、可在调用时指定
                                2、[推荐]也可在此方法中直接指定 将'XmindUitl-master'替换为当前项目名称即可（调用时即可直接调用 不用给参数）
        :param print_log: 是否打印日志信息
        :return: 指定项目的根路径
        """
        root_path = None
        project_path = os.path.abspath(os.path.dirname(__file__))
        # Windows
        if project_path.find('\\') != -1: 
            separator = '\\'
        # Mac、Linux、Unix
        if project_path.find('/') != -1: 
            separator = '/'

        root_path = project_path.split(separator)
        root_path =   separator.join(root_path[0:-1])+separator  
        # print(root_path)
        if isinstance(project_name,str) and project_name.strip()!="":
            project_name = project_name.strip()
            index = project_path.find(f'{project_name}{separator}')
            if index!= -1:
                root_path = project_path[:project_path.find(f'{project_name}{separator}') + len(f'{project_name}{separator}')]
        
        if print_log: 
            print(f'当前项目名称：{project_name}\r\n当前项目根路径：{root_path}')
        return root_path




def get_logger(logging_into_files=True,logging_dir=None,level="WARNING",tips=False,overwrite_logfiles=True):
    


    logger.remove(handler_id=None)
    if logging_into_files:
        if logging_dir is None:
            logging_dir =  ProjectUtil.project_root_path()

        if overwrite_logfiles:
            _logging_dir = logging_dir+'logs'
            if os.path.exists(_logging_dir): #FIXME:此处删除目录，不安全。最好是删除后缀为.log的文件
                import shutil
                shutil.rmtree(_logging_dir)

        logger.add(logging_dir+'logs/runtime_info.log',level="INFO",encoding='utf-8')
        logger.add(logging_dir+'logs/runtime_warn.log',level="WARNING",encoding='utf-8')
        logger.add(logging_dir+'logs/runtime_error.log',level="ERROR",encoding='utf-8')



    logger.add(sys.stdout, colorize=True,level=level)
    
    if tips:
        import torch
        print("\n")
        print(f"Print:using loguru as logger, and level= {level},log_dir={logging_dir+'logs'}")
        logger.info(f'torch.version={torch.__version__},has_gpu={torch.cuda.is_available()}')
        logger.warning((f'torch.version={torch.__version__},has_gpu={torch.cuda.is_available()}'))
        logger.error((f'torch.version={torch.__version__},has_gpu={torch.cuda.is_available()}'))
    return logger

root_path = ProjectUtil.project_root_path(project_name="YOLOV3")
logger = get_logger(logging_into_files=True,level="INFO",logging_dir=None,tips=False)
# logger = get_logger(logging_into_files=False,level="WARNING",logging_dir=None,tips=False)

# if __name__ == '__main__':
#     root_path = ProjectUtil.project_root_path(project_name="YOLOV3")
#     logger = get_logger(logging_into_files=False,level="WARNING",logging_dir=None,tips=False)