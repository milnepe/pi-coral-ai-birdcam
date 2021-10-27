#!/usr/bin/bash

#   Copyright 2019 Google LLC
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#	Birdcam runner
#	Adapted by Pete Milne from Coral Smart Bird Feeder Script.
#   Automates running the bird_classify code.


# Create a temp subdir in /tmp to store bird images and logs
tmp_dir=$(mktemp -d -t "birdcam-$(date +%Y-%m-%d)-XXXXXXXXXX")

echo "$tmp_dir"

cd birdcam || exit

python3 bird_classify.py \
	--model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
	--labels models/inat_bird_labels.txt \
	--top_k 1 \
	--threshold 0.4 \
	--storage "$tmp_dir" \
	--visit_interval 10  # Interval between bird visits in seconds
