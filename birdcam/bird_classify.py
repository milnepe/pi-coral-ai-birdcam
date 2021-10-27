#!/usr/bin/python3

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Coral Smart Bird Feeder

Adapted by Peter Milne

Uses ClassificationEngine from the EdgeTPU API to analyse birds in
camera video frames. Stores image of any bird visits and logs time and species.

Users define model, labels file, storage path, and
optionally can set this to training mode for collecting images for a custom
model.

"""
import argparse
import time
import logging
import threading
from PIL import Image

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters.classify import get_classes

import gstreamer


def save_data(image, results, path, ext='png'):
    """Saves camera frame and model inference results
    to user-defined storage directory."""
    tag = '%010d' % int(time.monotonic()*1000)
    name = '%s/img-%s.%s' % (path, tag, ext)
    image.save(name)
    print('Frame saved as: %s' % name)
    logging.info('Image: %s Results: %s', tag, results)


def print_results(start_time, last_time, end_time, results):
    """Print results to terminal for debugging."""
    inference_rate = ((end_time - start_time) * 1000)
    fps = (1.0/(end_time - last_time))
    print('\nInference: %.2f ms, FPS: %.2f fps' % (inference_rate, fps))
    for label, score in results:
        print(' %s, score=%.2f' % (label, score))


def do_training(results, last_results, top_k):
    """Compares current model results to previous results and returns
    true if at least one label difference is detected. Used to collect
    images for training a custom model."""
    new_labels = [label[0] for label in results]
    old_labels = [label[0] for label in last_results]
    shared_labels = set(new_labels).intersection(old_labels)
    if len(shared_labels) < top_k:
        print('Difference detected')
        return True
    return False


def user_selections():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='.tflite model path')
    parser.add_argument('--labels', required=True,
                        help='label file path')
    parser.add_argument('--videosrc', help='Which video source to use',
                        default='/dev/video0')
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='class score threshold')
    parser.add_argument('--storage', required=True,
                        help='File path to store images and results')
    parser.add_argument('--print', default=False, required=False,
                        help='Print inference results to terminal')
    parser.add_argument('--training', action='store_true',
                        help='Training mode for image collection')
    parser.add_argument('--visit_interval', action='store', type=int, default=2,
                        help='Minimum interval between bird visits')
    args = parser.parse_args()
    return args


def main():
    """Creates camera pipeline, and pushes pipeline through ClassificationEngine
    model. Logs results to user-defined storage. Runs either in training mode to
    gather images for custom model creation or capture mode that records images
    of bird visits if a model label is detected."""
    args = user_selections()
    print(args)
    print("Loading %s with %s labels." % (args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    input_tensor_shape = interpreter.get_input_details()[0]['shape']
    if (input_tensor_shape.size != 4 or
            input_tensor_shape[0] != 1):
        raise RuntimeError(
            'Invalid input tensor shape! Expected: [1, height, width, channel]')

    output_tensors = len(interpreter.get_output_details())
    if output_tensors != 1:
        raise ValueError(
            ('Classification model should have 1 output tensor only!'
             'This model has {}.'.format(output_tensors)))
    storage_dir = args.storage
    # Initialize logging file
    logging.basicConfig(filename='%s/results.log' % storage_dir,
                        format='%(asctime)s-%(message)s',
                        level=logging.DEBUG)
    last_time = time.monotonic()
    last_results = [('label', 0)]
    visitors = []

    DURATION = args.visit_interval
    timer = False

    def timed_event():
        nonlocal timer
        timer = True
        threading.Timer(DURATION, timed_event).start()

    timed_event()

    def user_callback(image, svg_canvas):
        nonlocal last_time
        nonlocal last_results
        nonlocal visitors
        nonlocal timer
        start_time = time.monotonic()
        common.set_resized_input(
            interpreter, image.size,
            lambda size: image.resize(size, Image.NEAREST))
        interpreter.invoke()
        results = get_classes(interpreter, args.top_k, args.threshold)
        end_time = time.monotonic()
        play_sounds = [labels[i] for i, score in results]
        results = [(labels[i], score) for i, score in results]
        if args.print:
            print_results(start_time, last_time, end_time, results)

        if args.training:
            if do_training(results, last_results, args.top_k):
                save_data(image, results, storage_dir)
        else:
            # Custom model mode:
            if len(results):
                visitor = results[0][0]
                if visitor not in EXCLUSIONS:
                    # If visit interval has past, clear visitors list
                    if timer:
                        print("next visit...")
                        visitors.clear()
                        timer = False
                    # If this is a new visit, add bird to visitors list
                    # so we don't keep taking the same image
                    if visitor not in visitors:
                        print("Visitor: ", visitor)
                        save_data(image,  visitor, storage_dir)
                        visitors.append(visitor)

        last_results = results
        last_time = end_time
    gstreamer.run_pipeline(user_callback, videosrc=args.videosrc)


if __name__ == '__main__':
    # Add to this list for false positives for your camera
    EXCLUSIONS = ['background',
                 'Branta canadensis (Canada Goose)']
    main()
