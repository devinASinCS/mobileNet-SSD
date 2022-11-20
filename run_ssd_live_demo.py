
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import time

if len(sys.argv) < 3:
    print(sys.argv)
    print('Usage: python run_ssd_live_example.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)

model_path = sys.argv[1]
label_path = sys.argv[2]

if len(sys.argv) >= 4:
    cap = cv2.VideoCapture(sys.argv[3])  # capture from file
    print('capture from file')
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)
    print('capture from camera')

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
print(f'Net classes num : {num_classes}')

net = create_mobilenetv3_ssd_lite(len(class_names)+1, is_test=True)
print('Create_mobilenetv3_ssd_lite !')

net.load(model_path)

predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200)
print('Create_mobilenetv3_ssd_lite_predictor !')


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
timer = Timer()
start_time = time.time()
counter = 0
fps = cap.get(cv2.CAP_PROP_FPS)
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[labels[i]-1], 4)

        cv2.putText(orig_image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 255, 255),
                    2)  # line type
    counter += 1
    if(time.time() - start_time) != 0:
        cv2.putText(orig_image, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))),
                    (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
        counter = 0
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    start_time = time.time()

cap.release()
cv2.destroyAllWindows()


