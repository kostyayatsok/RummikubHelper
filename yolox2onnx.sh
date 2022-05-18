# export MMDEPLOY_DIR=mmdeploy
# export MMDET_DIR=mmdetection
# python3 ${MMDEPLOY_DIR}/tools/deploy.py \
#     ${MMDEPLOY_DIR}/configs/mmdet/detection/single-stage_ncnn_static-416x416.py \
#     ${MMDET_DIR}/config.py \
#     tiles_yolox_tiny.pth \
#     images/coco-test-416/0011.jpg \
#     --work-dir work_dirs \
#     --device cpu \
#     --dump-info
# ./ncnn/build/tools/onnx/onnx2ncnn rummi.onnx rummi.param rummi.bin
# ./ncnn/build/tools/ncnnoptimize rummi.param rummi.bin rummi-opt.param rummi-opt.bin 65536
