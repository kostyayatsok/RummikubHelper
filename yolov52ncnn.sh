export model=../yolov5s_dist4
cd yolov5
python3 export.py --weights ${model}.pt --include torchscript --train
cd ../pnnx-20220428-ubuntu
./pnnx ${model}.torchscript inputshape=[1,3,640,640]
