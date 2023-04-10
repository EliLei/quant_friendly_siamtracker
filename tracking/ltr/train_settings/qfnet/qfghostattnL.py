from .qfconcat import _run, _default_settings


def run(settings):
    _default_settings(settings)
    settings.num_epochs=180
    settings.fusion_type = 'ghostattnL'
    settings.description = 'QFGHOSTATTNL'
    _run(settings)