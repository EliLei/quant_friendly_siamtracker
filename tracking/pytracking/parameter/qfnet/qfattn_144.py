from .qfconcat import _parameters

def parameters():
    params = _parameters('attn_144', False, template_size=144, search_size=304)

    params.template_size = 144
    params.template_factor = 2.0
    params.search_size = 304
    params.search_factor = params.template_factor * params.search_size / params.template_size

    return params
