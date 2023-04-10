from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.fusion_type = 'addbb'
    settings.description = 'QFADDBB'
    _run(settings)