
import numpy as np


observation_space_pre_dicts = {
    'high': np.array([
        12., 7., 24., 32.2,
        32.2, 32.2, 32.2, 100.,
        100., 100., 100., 1017.,
        1017., 1017., 1017., 953.,
        953., 953., 953., 0.282,
        8.0, 3.91, 1.0, 13.0,
        0.54, 0.54, 0.54, 0.54,
    ],
        dtype=np.float32),
    'low': np.array([
        1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.070,
        0.0, 0.0, -1.0, -8.91,
        0.21, 0.21, 0.21, 0.21
    ],
        dtype=np.float32),
    'shape': (28,),
    'dtype': 'float32'

}

if __name__ == "__main__":
    d = observation_space_pre_dicts
    print(d['high'].shape)
    print(d['low'].shape)
