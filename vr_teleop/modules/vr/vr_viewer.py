from .vr_base import VRViewerBase


class VRViewer(VRViewerBase):
    def __init__(self, visualize: bool = True):
        super().__init__(visualize=visualize)

    def __del__(self):
        del self.controllers
        del self.vr
        del self.renderer_context
        print("VRViewer deleted")
