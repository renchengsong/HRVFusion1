import numpy as np

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self

params = AttrDict(
    # Training params
    batch_size=64,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=30,
    n_specs=1,#fridge

    #mamba
    depth=8,
    embed_dim=128,
    mlp_ratio=2,
    drop_rate=0.,
    drop_path_rate=0.1,
    initializer_cfg=None,
    # Model params
    residual_channels=64,
    dilation_cycle_length=4,
    noise_schedule=np.linspace(1e-4, 0.35).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],
)
