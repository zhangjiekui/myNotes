import sys
from multiprocessing import freeze_support
import paddle
import torch
if __name__ == '__main__':
    if sys.platform.startswith('win'):
        # Hack for multiprocessing.freeze_support() to work from a
        # setuptools-generated entry point.
        freeze_support()

    # import paddle
    paddle.utils.run_check()
    # import torch
    print(torch.cuda.is_available())
    print(paddle.ones([3,3]))
