class Config:
    MAX_EPOCH = 10000
    BATCH_SIZE = 128
    g_noise_channel = 100

    g_base_channel = 64
    d_base_channel = 64

    image_size = (96, 96)
    image_channel = 3

    ABS_PATH = '/home/dell/data2/models/'
    train_d_frequence: int = 1
    train_g_frequence: int = 5

    g_lr: float = 0.00005
    d_lr: float = 0.00005

    clamp_value: float = 0.01

    images_folder = '/home/dell/data/Faces'
    device = 'cuda:0'


opt = Config()
