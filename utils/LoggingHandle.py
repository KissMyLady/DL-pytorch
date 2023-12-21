# Author: theOracle
# Datetime: 2022/10/3 11:18 PM
import logging
from logging.handlers import RotatingFileHandler
import os, sys
from pathlib import Path
import colorlog

log_colors_config = {
    'DEBUG': 'bold_cyan',
    'INFO': 'bold_green',
    'WARNING': 'bold_yellow',
    'ERROR': 'bold_red',
    'CRITICAL': 'red',
}


def LoggingHandle():
    """
    描述: 日志模块, 控制台与文件内同时记载
    注意: Flask开启调试模式会强制让logging=DEBUG
    :return: logger
    """
    logger = logging.getLogger()  # 实例化log对象
    logger.setLevel(logging.INFO)  # Log等级总开关
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    PATH = os.path.join(os.path.join(BASE_DIR, "deep-py"), "logs")
    if os.path.exists(PATH) is False:
        os.mkdir(PATH)
        pass

    file_path = os.path.join(PATH, "myPyProLog.log")

    if not logger.handlers:
        # 创建一个handler，用于输出到控制台
        ch_set3 = logging.StreamHandler()
        ch_set3.setLevel(logging.DEBUG)  # 输出到console的log等级的开关

        # formatter = logging.Formatter(fmt="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        """
        %(levelno)s: 打印日志级别的数值
        %(levelname)s: 打印日志级别名称
        %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
        %(filename)s: 打印当前执行程序名
        %(funcName)s: 打印日志的当前函数
        %(lineno)d: 打印日志的当前行号
        %(asctime)s: 打印日志的时间
        %(thread)d: 打印线程ID
        %(threadName)s: 打印线程名称
        %(process)d: 打印进程ID
        %(message)s: 打印日志信息
        """

        formatterFile = logging.Formatter(fmt="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s%(levelname)1.1s %(asctime)s %(reset)s| %(message_log_color)s%(levelname)-8s %(reset)s| %(log_color)s '
                '[%(filename)s%(reset)s:%(log_color)s%(module)s%(reset)s: '
                '%(log_color)s%(funcName)s%(reset)s:%(log_color)s%(''lineno)d] %(reset)s- %(white)s%(message)s',
            reset=True,
            log_colors=log_colors_config,
            secondary_log_colors={
                'message': {
                    'DEBUG': 'blue',
                    'INFO': 'blue',
                    'WARNING': 'blue',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red'
                }
            },
            style='%')  # 日志输出格式

        # 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限  Bytes字节 = 8位
        file_log_handler = RotatingFileHandler(file_path, maxBytes=1024 * 1024 * 5, backupCount=10, encoding="utf-8")

        ch_set3.setFormatter(formatter)  # 控制台 设置格式
        file_log_handler.setFormatter(formatterFile)  # 日志器 设置格式

        logger.addHandler(file_log_handler)   # 为全局的日志工具对象（flask app使用的）添加日记录器
        logger.addHandler(ch_set3)            # 将logger添加到handler里面

    # logger.removeHandler(file_log_handler)  # 写入完毕后, 要移除句柄, 否则重复写入
    return logger


def TestRun():
    """日志使用方法"""
    logger = LoggingHandle()

    # 日 志
    logger.debug('这是 logger debug message')
    logger.info('-' * 30)
    logger.info('这是 logger info message')
    logger.warning('这是 logger warning message')
    logger.error('这是 logger error message')
    logger.critical('这是 logger critical message')


if __name__ == "__main__":
    TestRun()