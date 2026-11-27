# Third-party notices

## TransWeather

Parts of the following files are derived from TransWeather
(https://github.com/jeya-maria-jose/TransWeather) via MWFormer
(https://github.com/taco-group/MWFormer):

- src/resilens/enhancement/mwformer_model/base_networks.py
- src/resilens/enhancement/mwformer_model/EncDec.py
- src/resilens/enhancement/mwformer_model/style_filter64.py

TransWeather is distributed under the MIT License:

MIT License

Copyright (c) 2021 Jeya Maria Jose

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## MWFormer

The same files contain modifications by the MWFormer authors
(Zhu, Tu, Liu, Bovik, Fan; IEEE TIP 2024). The MWFormer repository has no
licence; these modifications are used with no granted licence (see ADR-06).
They must not be included in a public release unless permission or a suitable
licence is obtained.

## YOLOX

src/resilens/detection/yolox_model.py is derived from YOLOX
(https://github.com/Megvii-BaseDetection/YOLOX), files
yolox/models/network_blocks.py, darknet.py, yolo_pafpn.py and yolo_head.py.

Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
A copy of the License is in LICENSES/Apache-2.0.txt and at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications for Resilens: inference-only copy; training losses, the
Darknet53/ResNet backbones and configurable activations are removed.
Module and parameter names are kept so official checkpoints load unchanged.

The pretrained weights (weights/yolox_nano_coco.pth) are not stored in this
repository; they are downloaded from the official YOLOX release.
