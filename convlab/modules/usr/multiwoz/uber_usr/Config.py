class Config:
    def __init__(self):
        self.goal_embedding_size = 100
        self.usr_embedding_size = 100
        self.sys_embedding_size = 100
        self.dropout = 0.5
        self.layer_num = 3
        self.hidden_state_num = 128
        self.lr = 0.001
        self.lr_decay = 0.5

config = Config()