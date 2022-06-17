class BaseConfig:
    n_residual_blocks: int = 9
    input_nc_A: int = 3
    input_nc_B: int = 3
    n_D_layers: int = 4
    device: str = 'cuda:1'
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    img_height: int = 256
    img_width: int = 256
    lambda_cyc: float = 10
    lambda_id: float = 0.5
    train_d_fre: int = 1
    train_g_fre: int = 1
    BATCH_SIZE: int = 1
    MAX_EPOCH: int = 10000
    ABS_PATH: str = '/home/dell/data2/models/'
    data_set_root: str = '/home/dell/data/Cat/grumpifycat/'




