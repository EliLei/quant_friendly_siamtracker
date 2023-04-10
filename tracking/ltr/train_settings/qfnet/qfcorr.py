from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.fusion_type = 'corr'
    settings.description = 'QFCORR'
    _run(settings)