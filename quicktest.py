import yolo
import torch

p = torch.zeros(1, 3, 13, 13, 7)
p[0, 0, 1, 1] = torch.tensor([0.5, 0.5, -1.2879, -1.0341, 1.0, 0.0, 1.0])

t = torch.tensor([
    [0, 1, 1/13, 1/13, 2/13, 2/13],
]).float()

anchors_13 = [(116/32, 90/32), (156, 198), (373, 326)]
d = yolo.YOLODetector(anchors_13, 2)
d.stride = 32
l = d.loss(p, t)
print(l)
