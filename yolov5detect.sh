# export MODEL="../yolov5s_dist4.pt"
export MODEL="../multilabel_yolov5n6.pt"
export SOURCE="../images/unlabeld_test/"
cd yolov5-my
python3 detect.py --weights $MODEL  --source  $SOURCE --line-thickness 1 --conf-thres 0.5 --imgsz 1280 #  --save-crop
