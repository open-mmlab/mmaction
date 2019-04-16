from .base import BaseRecognizer

class TSN2D(BaseRecognizer):

    def __init__(self,
                 backbone,
                 tempon):

        super(TSN2D, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        