class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, info):
        with open(self.log_path, 'a+') as f:
            f.write(info)
            f.write('\n')
        f.close()

    def clear(self):
        with open(self.log_path, 'w+') as f:
            f.write('')
        f.close()