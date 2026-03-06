import time
import torch
import torch.nn.functional as F
import torch.nn as nn

# Import your YOLO model
from models.yolo import *
model = Model() # Replace with your actual YOLOv5 model instantiation

def find_layer_by_name(model, layer_name):
    hierarchy = layer_name.split('.')
    target_layer = model
    for h in hierarchy:
        if h.isdigit():
            # Handle indexing for modules that have lists
            target_layer = target_layer[int(h)]
        else:
            target_layer = getattr(target_layer, h, None)
            if target_layer is None:
                return None

    if isinstance(target_layer, nn.Conv2d):
        print(f"Selected layer '{layer_name}' is a Conv2d layer.")
        return target_layer
    else:
        return None

class YOLOV5GradCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.layer_name = layer_name
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            print(f'Backward Hook - Module: {module.__class__.__name__}')
            self.gradients[self.layer_name] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations[self.layer_name] = output
            return None

        target_layer = find_layer_by_name(self.model, layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

        print(f'Layer name: {layer_name}')
        print('Target Layer:', target_layer)
        print('Target Layer Type:', type(target_layer))
        print(layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        print('Target Layer- ', target_layer)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        tic = time.time()
        preds, logits = self.model(input_img)
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            tic = time.time()
            score.backward(retain_graph=True)
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            gradients = self.gradients[self.layer_name]

            activations = self.activations[self.layer_name]

            b, k, u, v = gradients.size()
            assert gradients.size(1) == activations.size(1), "Number of channels in gradients and activations must match"
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds

    def __call__(self, input_img):
        return self.forward(input_img)

# Instantiate YOLOV5GradCAM with the desired layer name
#saliency_method = YOLOV5GradCAM(model=model, layer_name='model.23.cv3_act', img_size=(640, 640))
