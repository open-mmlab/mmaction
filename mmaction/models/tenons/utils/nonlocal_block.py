from ..spatial_temporal_modules.non_local import NonLocalModule


def build_nonlocal_block(cfg):
    """ Build nonlocal block

    Args:
    """
    assert isinstance(cfg, dict)
    cfg_ = cfg.copy()
    return NonLocalModule(**cfg_)
