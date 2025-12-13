import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']  #动态地扫描并导入所有位于 archs 目录下的以 _arch.py 结尾的模块，然后通过配置选项 opt 创建一个网络对象

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__)) #获取当前文件所在的目录路径。
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
#扫描当前目录下的所有文件，筛选出以 _arch.py 结尾的文件，并去掉文件扩展名，得到文件名列表
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt) #深拷贝，不修改原配置。尽可能的保留信息
    network_type = opt.pop('type') #从配置中提取网络信息，移除Type键
    net = ARCH_REGISTRY.get(network_type)(**opt) #使用 opt 配置字典中的其他项作为参数来实例化网络。
    logger = get_root_logger() #获取日志对象，并记录网络的创建信息，输出网络的类名。
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
#构建并返回network实例