python -u main.py --pretrained_weights  /media/nnir712/F264A15264A119FD/zzh/detect/yolo/PyTorch-YOLOv3/weights/darknet53.conv.74 \
                  --cfg config/yolov3.cfg \
                  --data config/yolov3.data \
                  --batch_size 4 \
                  --test_batch_size 4 \
                  | tee logs/yolov3.log
