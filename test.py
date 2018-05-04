__author__ = 'ww'
from dqn import config

def main():
    cf_ = config.Config()
    print(cf_.history_length)


if __name__ == "__main__":
    main()