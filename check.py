import yolo
from loading_weights import load_model_parameters
import os
import torch
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize
from dataset import YOLODataset
from nms import reduce_boxes
import dataset
import yolo_post

from primitive_dataloader import PrimitiveDataModule


def get_img_transform():
    return Compose([Lambda(dataset.resize_with_respect), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_target_transform():
    return dataset.resize_bbs

if __name__ == "__main__":
    WEIGHTS_DIR = 'weights'
    path_to_darknet = os.path.join(WEIGHTS_DIR, 'darknet53.conv.74')
    path_to_yolo = os.path.join(WEIGHTS_DIR, 'yolov3.weights')

    model = yolo.YOLOv3(80, 3)

    load_model_parameters(path_to_yolo, model)

    #train_path_anno = 'test-new/annotations'
    #train_path_img = 'test-val2017'
    #yolo_dataset = YOLODataset(train_path_anno, train_path_img, transform=get_img_transform(), target_transform=get_target_transform())

    data_model = PrimitiveDataModule(None, 'test-val2017', batch_size=1, num_workers=4, img_transform=get_img_transform())
    data_model.setup()


    img_size = 416
    num_classes = 80
    obj_coeff, noobj_coeff = 1, 100
    ignore_threshold = 0.5

    proc_52 = yolo_post.YOLOProcessor([(10, 13), (16, 30), (33, 23)], 8, img_size, num_classes, obj_coeff, noobj_coeff, ignore_threshold)
    proc_26 = yolo_post.YOLOProcessor([(30, 61), (62, 45), (59, 119)], 16, img_size, num_classes, obj_coeff, noobj_coeff, ignore_threshold)
    proc_13 = yolo_post.YOLOProcessor([(116, 90), (156, 198), (373, 326)], 32, img_size, num_classes, obj_coeff, noobj_coeff, ignore_threshold)

    for i, batch in enumerate(data_model.predict_dataloader()):
        if i == 0:
            x, _, path, _ = batch
            x, path = x[0], path[0]
            x = x.unsqueeze(0)
            print(path)
            y, loss = model(x, None)
            #
            y13, y26, y52 = y
            y13, y26, y52 = proc_13.reshape_and_sigmoid(y13), proc_26.reshape_and_sigmoid(y26), proc_52.reshape_and_sigmoid(y52)
            y13, y26, y52 = proc_13.process_after_loss(y13), proc_26.process_after_loss(y26), proc_52.process_after_loss(y52)
            y = torch.cat([y13, y26, y52], dim=1)
            #
            z = reduce_boxes(y)
            print(z)
            dataset.visualize_results(path, z.tolist())
