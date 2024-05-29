import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import onnx
import torch

# Custom transform to handle 6 channels
class CustomTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(CustomTransform, self).__init__(min_size, max_size, image_mean, image_std)

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]





def model_init():
    num_classes = 2
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    # Define mean and std for 6 channels
    image_mean = [0.485, 0.456, 0.406, 0.9809]  # Example: same as RGB for extra channels
    image_std = [0.229, 0.224, 0.225, 0.156175]    # Example: same as RGB for extra channels

    # Create a custom transform with modified mean and std for 6 channels
    transform = CustomTransform(min_size=(800,), max_size=1333, image_mean=image_mean, image_std=image_std)
    model.transform = transform
    modified_conv = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # with torch.no_grad():
    #     modified_conv.weight[:, :3, :, :] = model.backbone.body.conv1.weight
    #     modified_conv.weight[:, 3:, :, :] = model.backbone.body.conv1.weight[:, :3, :, :]

    model.backbone.body.conv1 = modified_conv


    return model





# def main():
#     model = model_init()
#     model.eval()
#     x = torch.randn(1, 4, 800, 800)
#     print(model(x))
#     # torch.onnx.export(model, x, "model.onnx", verbose=True, opset_version=11)


# if __name__ == "__main__":
#     main()

