from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.fusion_type = 'concatbb'
    settings.description = 'QFCONCATBB'
    _run(settings)