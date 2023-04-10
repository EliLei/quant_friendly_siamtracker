from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.fusion_type = 'ghostattn1'
    settings.description = 'QFGHOSTATTN1'
    _run(settings)