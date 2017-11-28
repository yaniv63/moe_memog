import logging

def get_logger(path):
    # add logging
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s - %(module)s - %(processName)s - %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=path+'run.log')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    # console = logging.FileHandler(filename=path+'run.log',mode='w')
    # console.setLevel(logging.DEBUG)
    # # set a format which is simpler for console use
    # formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s - %(module)s - %(processName)s -  %(message)s')
    # # tell the handler to use this format
    # console.setFormatter(formatter)
    # # add the handler to the root logger
    logger = logging.getLogger('root')
    # if len(logger.handlers) > 0:
    #     logger.removeHandler(logger.handlers[0])
    # logger.addHandler(console)
    return logger

