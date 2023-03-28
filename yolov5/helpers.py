from pathlib import Path

from yolov5.models.common import AutoShape, DetectMultiBackend
from yolov5.utils.general import LOGGER, logging
from yolov5.utils.torch_utils import torch


def load_model(model, model_type, device=None, autoshape=True, verbose=False):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str|bytes): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    # set logging
    if not verbose:
        LOGGER.setLevel(logging.WARNING)

    # set device if not given
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif type(device) is str:
        device = torch.device(device)

    model = DetectMultiBackend(weights=model, weights_type=model_type, device=device)

    if autoshape:
        model = AutoShape(model, model_type)  # for file/URI/PIL/cv2/np inputs and NMS
    return model.to(device)


if __name__ == "__main__":
    model_path = "yolov5/weights/yolov5s.pt"
    device = "cuda"
    model = load_model(model_path=model_path, config_path=None, device=device)

    from PIL import Image
    imgs = [Image.open(x) for x in Path("yolov5/data/images").glob("*.jpg")]
    results = model(imgs)
