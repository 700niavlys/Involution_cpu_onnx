Involution_CPU_ONNX_CAFFE
====================
A huge thanks to their excellent work By Duo Li, Jie Hu, Changhu Wang, Xiangtai Li, Qi She, Lei Zhu, Tong Zhang, and Qifeng Chen<br>
Inspired by CVPR2021 Involution, a realization of pytorch Involution with ONNX transformation and quantization.<br>
I have no GPU in my personal laptop, so I have to choose to install cpu version for all.<br>

Article: 
----------
Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255) (CVPR'21)<br>
involution is a general-purpose neural primitive that is versatile for a spectrum of deep learning models on different vision tasks. <br>
involution bridges convolution and self-attention in design, while being more efficient and effective than convolution, simpler than self-attention in form.

Installation
------------
Suppose you have mmdetection installed( In my case, I installed mmdetection cpu version in Ubuntu 20.04)<br>
git clone https://github.com/open-mmlab/mmdetection # and install<br>

Then clone Involution official implementation of Involution<br>
git clone https://github.com/d-li14/involution<br>

After that, cp `model` and `config` files from Involution to mmdetection<br>
cp involution/det/mmdet/models/backbones/* mmdetection/mmdet/models/backbones<br>
cp involution/det/mmdet/models/necks/* mmdetection/mmdet/models/necks<br>
cp involution/det/mmdet/models/utils/* mmdetection/mmdet/models/utils<br>

cp involution/det/configs/_base_/models/* mmdetection/mmdet/configs/_base_/models<br>
cp involution/det/configs/_base_/schedules/* mmdetection/configs/_base_/schedules<br>
cp involution/det/configs/involution mmdetection/configs -r<br>

Test
-----
In my case, I choosed the Object detection task on COCO:<br>
`RedNet-50-FPN` with:<br>
faster_rcnn_red50_neck_fpn_1x_coco.py<br>
_base_ = [<br>
    '../_base_/models/faster_rcnn_red50_neck_fpn.py',<br>
    '../_base_/datasets/coco_detection.py',<br>
    '../_base_/schedules/schedule_1x_warmup.py', '../_base_/default_runtime.py'<br>
]<br>
and the pre-trained model:<br>
~118M<br>
https://hkustconnect-my.sharepoint.com/personal/dlibh_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdlibh%5Fconnect%5Fust%5Fhk%2FDocuments%2Finvolution%2Fdet%2Ffrcnn%5Fneck%2D8dbf2cd5%2Epth&parent=%2Fpersonal%2Fdlibh%5Fconnect%5Fust%5Fhk%2FDocuments%2Finvolution%2Fdet&originalPath=aHR0cHM6Ly9oa3VzdGNvbm5lY3QtbXkuc2hhcmVwb2ludC5jb20vOnU6L2cvcGVyc29uYWwvZGxpYmhfY29ubmVjdF91c3RfaGsvRVY5MHN0QUpJWHhFbkRSZTBRTTBsdndCX2ptOWp3cXdIb0JPVlZPcW9zUEhKdz9ydGltZT1sUFRYajFYczJFZw<br>

Finally, run a `demo`:<br>

from mmdet.apis import init_detector, inference_detector<br>
config_file = 'your_rednet_config_file.py'<br>
checkpoint_file = 'your_rednet_model.pth'<br>
device = 'cpu'<br>
model = init_detector(config_file, checkpoint_file, device=device)<br>
inference_detector(model, 'demo/demo.jpg')<br>

then, <br>
       show_result_pyplot(model, args.img, result, score_thr=args.score_thr)<br>
we can compare the result between RedNet and Normal Convolution by one-time inference.<br>

Visualization network arch
-----
Using torch.save to save both `network` and `weight` together<br>
~119M, we see by adding the network arch into the model file, the size augmented a little<br>
If I try to use `Netron` to load the saved model for now, it only shows the` meta graph` of the model as `backbone`,`neck`,etc, I cannot see the detailed graph with each layer<br>
So I choose to transform it to `onnx model` and then reload the model to Netron to see the details.

Visualization each feature map and kernel
-----
pip install flashtorch<br>
pip install torchviz with graphviz<br>
others:<br>
torchsummaryX<br>
etc.<br>
then I can see deeply why Involution works and how does the information exchanged between spatial space and channel space<br>
pay attention to kernel size 1x1,3x3 and `7x7`, that's how it realized model long range interaction by lowing cost.<br>

A tip: pay attention to a parameter `img_metas` if you encounter problem, it stoped me from using forward function<br>
Solution:rewrite yourself a forward function or, set a fake img_metas to forward.


ONNX transform
---------------
python pytorch2onnx.py \\<br>
    your_config_file.py \\<br>
    your_model_file.pth \\<br>
    --output-file your_output_fileonnx \\<br>
    --input-img your_demo_img.jpg \\<br>
    --verify<br>
After that, we see the size of ONNX model keeps the same as before<br>
Use Netron to see the graph,such b-e-a-utiful<br>

ONNX quantization
---------------
install onnx runtime with torch==0.4.0, a new conda env<br>
using onnx to quantize your model with size 119M<br>
In my case, I choosed a simple but fast quantization method,<br>
Final size: ~73M<br>

ONNX to CAFFE transformatiom
---------------
Nothing more to say,just pay attention to unsqueeze layer<br>
Have fun :) <br>
