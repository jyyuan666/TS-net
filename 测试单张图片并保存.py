from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  # , show_result_pyplot
import cv2


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    return img


def main():
    # config文件
    config_file = '/home/yuan3080/桌面/detection_paper_6/mmdetection-master1/mmdetection-master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # 训练好的模型
    checkpoint_file = '/home/yuan3080/桌面/detection_paper_6/mmdetection-master1/mmdetection-master/configs/faster_rcnn_log_faster_rcnn_r50_fpn_1x_coco/latest.pth'

    # model = init_detector(config_file, checkpoint_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 图片路径
    name = '/home/yuan3080/桌面/detection_paper_6/mmdetection-master1/mmdetection-master/data/VOCdevkit/VOC2007/JPEGImages/001.jpg'
    # 检测后存放图片路径
    out_dir = './configs/faster_rcnn_log_faster_rcnn_r50_fpn_1x_coco'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    result = inference_detector(model, name)
    img = show_result_pyplot(model, name, result, score_thr=0.8)
    # 命名输出图片名称
    cv2.imwrite("{}/{}.jpg".format(out_dir, 122), img)


if __name__ == '__main__':
    main()