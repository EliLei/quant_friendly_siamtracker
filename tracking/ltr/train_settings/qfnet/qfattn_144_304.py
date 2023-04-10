from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.search_area_factor = {'template': 2., 'search': 304 * 2 / 144}
    settings.output_sz = {'template': 144, 'search': 304}
    settings.feature_sz = 19
    settings.fusion_type = 'attn_144'
    settings.description = 'QFATTN144'
    _run(settings)