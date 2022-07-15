import os
import re
import sys
import cv2
import math
import onnx
import pickle
import platform
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from typing import Union


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_PATH, 'input')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
MODEL_PATH  = os.path.join(BASE_PATH, 'model')

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def get_image(path: str) -> np.ndarray:
    return cv2.cvtColor(src=cv2.imread(path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)


def show_image(image: np.ndarray, cmap: str="gnuplot2", title: str=Union[str, None]) -> None:
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


class Model(object):
    def __init__(self) -> None:
        self.ort_session = None
        self.size: int = 416
        self.path: str = os.path.join(MODEL_PATH, 'model.onnx')
        with open(os.path.join(MODEL_PATH, "classes.pkl"), "rb") as fp: self.classes = pickle.load(fp)
        ort.set_default_logger_severity(3)
    
    def setup(self) -> None:
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        scale = min(self.size / w, self.size / h)
        # nh, nw = math.ceil(h * scale), math.ceil(w * scale)

        nh, nw = int(h * scale), int(w * scale)
        hh: int = (self.size - nh) // 2
        ww: int = (self.size - nw) // 2

        image = cv2.resize(src=image, dsize=(nw, nh), interpolation=cv2.INTER_AREA)
        new_image = np.ones((self.size, self.size, 3), dtype=np.uint8) * 128
        
        if abs(nh-(self.size - 2*hh)) == 1: new_image[hh:self.size-hh-1, ww:self.size-ww, :] = image
        elif abs(nw-(self.size - 2*ww)) == 1: new_image[hh:self.size-hh, ww:self.size-ww-1, :] = image
        else: new_image[hh:self.size-hh, ww:self.size-ww, :] = image

        new_image = new_image.transpose(2, 0, 1).astype("float32")
        new_image /= 255
        new_image = np.expand_dims(new_image, axis=0)
        return new_image
    
    def infer(self, image: np.ndarray) -> tuple:
        
        image_h, image_w, _ = image.shape
        image = self.preprocess(image=image)

        input = {
            self.ort_session.get_inputs()[0].name : image,
            self.ort_session.get_inputs()[1].name : np.array([image_h, image_w], dtype=np.float32).reshape(1, 2),
        }
        
        boxes, scores, indices = self.ort_session.run(None, input)

        out_boxes, out_scores, out_classes = [], [], []

        if len(indices[0]) != 0:
            for idx_ in indices[0]:
                out_classes.append(idx_[1])
                out_scores.append(scores[tuple(idx_)])
                idx_1 = (idx_[0], idx_[2])
                out_boxes.append(boxes[idx_1])
            
            x1, y1, x2, y2 = int(out_boxes[0][1]), int(out_boxes[0][0]), int(out_boxes[0][3]), int(out_boxes[0][2])
            
            return self.classes[out_classes[0]], out_scores[0], (x1, y1, x2, y2)
        else:
            return None, None, None


def main():
    args_1: tuple = ("--mode", "-m")
    args_2: tuple = ("--filename", "-f")
    args_3: tuple = ("--downscale", "-ds")
    args_4: tuple = ("--save", "-s")

    mode: str = "image"
    filename: str = "Test_1.jpg"
    downscale: float = None
    save: bool = False

    if args_1[0] in sys.argv: mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: mode = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: filename = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: filename = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_3[0]) + 1])
    if args_3[1] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_3[1]) + 1])

    if args_4[0] in sys.argv or args_4[1] in sys.argv: save = True

    model = Model()
    model.setup()

    if re.match(r"image", mode, re.IGNORECASE):
        image = get_image(os.path.join(INPUT_PATH, filename))
        disp_image = image.copy()
        label, score, box = model.infer(image)

        if label is not None:
            if not save:
                cv2.rectangle(disp_image, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(0, 255, 0), thickness=2)
                show_image(image=disp_image, title=f"{label} ({score:.2f})")
            else:
                cv2.imwrite(
                    os.path.join(OUTPUT_PATH, filename[:-4] + f" - ROI ({label}).jpg"), 
                    cv2.cvtColor(src=disp_image[box[1]:box[3], box[0]:box[2]], code=cv2.COLOR_RGB2BGR)
                )
        else:
            breaker()
            print("No Detections")
            breaker()
    
    elif re.match(r"video", mode, re.IGNORECASE):
        cap = cv2.VideoCapture(os.path.join(INPUT_PATH, filename))
    
        while True:
            ret, frame = cap.read()
            if ret:
                if downscale:
                    frame = cv2.resize(src=frame, dsize=(int(frame.shape[1]/downscale), int(frame.shape[0]/downscale)), interpolation=cv2.INTER_AREA)
                disp_frame = frame.copy()
                label, score, box = model.infer(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)) 
                if label is not None:
                    cv2.rectangle(disp_frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(0, 255, 0), thickness=2)
                # else:
                #     cv2.putText(disp_frame, text="No Detections", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                
                cv2.imshow("Feed", disp_frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif re.match(r"realtime", mode, re.IGNORECASE):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: break
            disp_frame = frame.copy()
            label, score, box = model.infer(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)) 
            if label is not None:
                cv2.rectangle(disp_frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(0, 255, 0), thickness=2)
            # else:
            #     cv2.putText(disp_frame, text="No Detections", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            
            cv2.imshow("Feed", disp_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()

    else:
        breaker()
        print("--- Unknown Mode ---".upper())
        breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)