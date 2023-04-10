from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.fusion_type = 'ghostattn2'
    settings.description = 'QFGHOSTATTN2'
    _run(settings)