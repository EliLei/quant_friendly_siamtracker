from .qfconcat import _run

def run(settings):
    settings.fusion_type = 'attn1.5'
    settings.description = 'QFATTN1.5'
    _run(settings)