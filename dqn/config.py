
import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read('./dqn/config.dat')

class Config(object):

    histoty = cf.get('input','history')
    history_length = cf.getint('input','history_length')
    screen_height = cf.getint('input','screen_height')
    screen_width = cf.getint('input','screen_width')
    max_epoch = cf.getint('input','max_epoch')
    max_step = cf.getint('input','max_step')
    learn_start = cf.getint('input', 'learn_start')
    train_frequency = cf.getint('input', 'train_frequency')
    test_step = cf.getint('input', 'test_step')
    ep_end_t = cf.getint('input','ep_end_t')
    target_q_update = cf.getint('input','target_q_update')
    GAMMA = cf.getfloat('input', 'GAMMA')

    memory_size = cf.getint('model','memory_size')
    batch_size = cf.getint('model','batch_size')
    replace_counter = cf.getint('model', 'replace_counter')

if __name__ == "__main__":
    config = Config()
    print ("1",config.histoty)
'''
import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read('config.dat')

class Config(object):
    histoty = cf.get('input','history')

if __name__ == "__main__":
    config = Config()
    print("1", config.histoty)

'''