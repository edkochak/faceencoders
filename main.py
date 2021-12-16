import PIL
import torch.nn as nn
from torchvision import models
import torch
import cv2
import numpy as np
from PIL import Image
import os
import test as t
import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
import scipy.spatial.distance as distance
def rand(): return random.randint(100, 255)


def get_net(path):
    model = t.Net()
    model.pretrain(embad=50)
    model.load_state_dict(torch.load(path))
    return model


def main(path_to_net, filename_capture, record=False, two_cascades=False):
    transform = T.Compose([
        T.Resize([60, 60]),
        T.ToTensor()
    ])
    tb = SummaryWriter()
    last = []
    lastimage = []
    countclass = []
    threshold = 0.3
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
    if two_cascades:
        face_cascade1 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml')
    d = 0.2
    N = 200

    if filename_capture == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(filename_capture)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 40500)  # Берем не начало

    if record:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
            *'mp4v'), 25.0, (frame_width, frame_height))

    model = get_net(path_to_net)

    model.eval()
    count = 0
    vec1 = []
    meta = []
    while True:
        ret, original_image = cap.read()
        count += 1

        h1, w1, _ = original_image.shape
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        detected_faces = list(face_cascade.detectMultiScale(grayscale_image))
        if two_cascades:
            detected_faces += list(face_cascade1.detectMultiScale(grayscale_image))
        imgs = torch.Tensor()
        if len(detected_faces) > 0:
            img_faces = []
            # Проходимся по всем лицам
            for faces in detected_faces:
                (column, row, width, height) = faces

                # Отступы
                dx = int(d*width)
                dy = int(d*height)
                # Обрезка
                face_image = original_image[max(row-dy, 0):min(row +
                                                               height+dy, h1), max(column-dx, 0):min(column+width+dx, w1), ::-1]
                img = Image.fromarray(face_image)

                img = transform(img)
                img_faces.append(img)

                if 0 in imgs.shape:
                    # если первая фотка, то создаем пачку
                    imgs = img.unsqueeze(0)
                else:
                    # Добавляем в пачку
                    imgs = torch.cat((imgs, img.unsqueeze(0)), dim=-1)

            predict = torch.nn.functional.normalize(model(imgs))
            dx1 = 0
            dy1 = 0

            # Лица на фотке
            for n, face in enumerate(predict):
                # Тензор в numpy и добавление в Tensorboard
                face = face.detach().numpy()
                vec1.append((face, img_faces[n].numpy()))

                # Поиск ближайших соседей
                if last == []:
                    sum_res = []
                else:
                    d1 = distance.cdist([i[1] for i in last], [
                                        face], 'minkowski', p=2.)
                    sum_res = [(last[n][0], i[0]) for n, i in enumerate(d1)]
                    sum_res = sorted(
                        sum_res, key=lambda x: x[1])[:N]

                # Голосование
                voices = {}
                for id, w in sum_res:
                    if id not in voices:
                        voices[id] = 0
                    voices[id] += np.e**(-w**2)

                # if sum_res:
                #     print(sum_res[0][1])

                # Если порог не пройден, то это новый человек
                if sum_res == [] or (sum_res != [] and sum_res[0][1] > threshold):
                    if sum_res:
                        print(sum_res[0][1])
                    color = (rand(), rand(), rand())
                    last.append((len(last), face))
                    (column, row, width, height) = detected_faces[n]
                    dx = int(d*width)
                    dy = int(d*height)
                    face_image = original_image[max(row-dy, 0):min(row +
                                                                   height+dy, h1), max(column-dx, 0):min(column+width+dx, w1)]
                    lastimage.append((face_image, color))
                    maxkey = len(lastimage)-1

                else:
                    maxkey = max(voices, key=lambda x: voices[x])
                    face_image, color = lastimage[maxkey]

                meta.append(maxkey)

                # Обрезка и добавление фотки в левый вверхний угол, а также рамка
                face_image = cv2.resize(
                    face_image, (100, 100), interpolation=cv2.INTER_AREA)
                img_height, img_width, _ = face_image.shape
                original_image[dx1:img_height, dy1:img_width] = face_image
                dx1 += 100
                dy1 += 100
                start_point, end_point = (max(
                    column-dx, 0), max(row-dy, 0)), (min(column+width+dx, w1), min(row + height+dy, h1))
                original_image = cv2.rectangle(
                    original_image, start_point, end_point, color, thickness=6)

        if record:
            out.write(original_image)
        cv2.imshow("camera", original_image)
        if cv2.waitKey(1) == 27:
            break

    # Преобразуем данные, удобные Tensorboard
    imgs1 = [i[1] for i in vec1]
    vec1 = [i[0] for i in vec1]
    vec1 = torch.from_numpy(np.array(vec1))
    imgs1 = torch.from_numpy(np.array(imgs1))

    print(vec1.shape, imgs1.shape)
    tb.add_embedding(vec1, label_img=imgs1, metadata=meta)
    tb.close()

    cap.release()
    if record:
        out.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    # test()
    main()
