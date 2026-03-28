# ========== Helper to handle model outputs ==========
def unwrap_generator_output(g_out):
    """
    Accepts generator output which may be:
      - sr_img (tensor)
      - (sr_img, mu, sigma)
    Returns (sr_img, mu, sigma) where mu/sigma may be None if absent.
    """
    if isinstance(g_out, (tuple, list)):
        if len(g_out) == 3:
            return g_out[0], g_out[1], g_out[2]
        elif len(g_out) == 2:
            return g_out[0], None, g_out[1]
        else:
            return g_out[0], None, None
    else:
        return g_out, None, None