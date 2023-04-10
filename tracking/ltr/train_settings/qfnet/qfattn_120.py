from .qfconcat import _run

def run(settings):
    settings.fusion_type = 'attn1.2'
    settings.description = 'QFATTN1.2'
    _run(settings)