import cv2
import numpy as np
import torch
import time
import argparse
from PIL import Image, ImageDraw, ImageFont
from utils.util import *
from MTCNN import create_mtcnn_net

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MTCNN Video')
    parser.add_argument("--scale", dest='scale', help=
    "input frame scale to accurate the speed", default="0.1", type=float)
    parser.add_argument('--mini_face', dest='mini_face', help=
    "Minimum face to be detected. derease to increase accuracy. Increase to increase speed",
                        default="32", type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(0)
    while True:
        isSuccess, frame = cap.read()
        if isSuccess:
            try:
                start_time = time.time()
                input = resize_image(frame, args.scale)
                bboxes, landmarks = create_mtcnn_net(input, args.mini_face, device, p_model_path='weights/pnet_Weights', r_model_path='weights/rnet_Weights', o_model_path='weights/onet_Weights')

                if bboxes != []:
                    bboxes = bboxes / args.scale
                    landmarks = landmarks / args.scale

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype('utils/simkai.ttf', 30)

                FPS = 1.0 / (time.time() - start_time)
                draw.text((10, 10), 'FPS: {:.1f}'.format(FPS), fill=(0, 0, 0), font=font)

                for i, b in enumerate(bboxes):
                    draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline='blue', width=5)

                for p in landmarks:
                    for i in range(5):
                        draw.ellipse([(p[i] - 4.0, p[i + 5] - 4.0), (p[i] + 4.0, p[i + 5] + 4.0)], outline='yellow')

                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            except:
                print('detect error')

            cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()