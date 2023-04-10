from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.fusion_type = 'attn'
    settings.description = 'QFATTN'
    _run(settings)