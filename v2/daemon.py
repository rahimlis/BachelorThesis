# Copyright (C) 2017 DataArt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import threading
import logging.config
import datetime
import numpy as np
from collections import deque
from scipy.io import wavfile
from devicehive_webconfig import Server, Handler
import numpy as np
import tensorflow as tf
import torch
import operator
from v2 import params
from v2.train_network import get_model, transpose_features
from v2 import feat_extractor as extractor
from v2 import utils
from v2.audio.captor import Captor
from v2.audio.processor import WavProcessor, format_predictions

logger = logging.getLogger('audio_analysis.daemon')


class DeviceHiveHandler(Handler):
    _device = None

    def handle_connect(self):
        self._device = self.api.put_device(self._device_id)
        super(DeviceHiveHandler, self).handle_connect()

    def send(self, data):
        if isinstance(data, str):
            notification = data
        else:
            try:
                notification = json.dumps(data)
            except TypeError:
                notification = str(data)

        self._device.send_notification(notification)


class Daemon(Server):
    _process_thread = None
    _process_buf = None
    _ask_data_event = None
    _shutdown_event = None
    _captor = None
    _sample_rate = 16000
    _processor_sleep_time = 0.01

    events_queue = None

    def __init__(self, *args, **kwargs):
        min_time = 5
        max_time = 5
        self._save_path = None

        super(Daemon, self).__init__(*args, **kwargs)

        self.events_queue = deque(maxlen=10)
        self._ask_data_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._process_thread = threading.Thread(target=self._process_loop,
                                                name='processor')
        self._process_thread.setDaemon(True)

        self._captor = Captor(min_time, max_time, self._ask_data_event,
                              self._process, self._shutdown_event)

    def _start_capture(self):
        logger.info('Start captor')
        self._captor.start()

    def _start_process(self):
        logger.info('Start processor loop')
        self._process_thread.start()

    def _process(self, data):
        self._process_buf = np.frombuffer(data, dtype=np.int16)

    def _on_startup(self):
        self._start_process()
        self._start_capture()

    def _on_shutdown(self):
        self._shutdown_event.set()

    def format_top5(self, class_map, top5):
        result = []
        probabilities, predictions = top5.values.reshape(-1), top5.indices.reshape(-1)
        dict = {}
        for cl, p in zip(predictions, probabilities):
            dict[class_map[cl]] = round(p, 6)

        sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

        for tupl in sorted_dict:
            cl, prob = tupl
            result.append(cl + ": " + str(prob))
        return result

    def _process_loop(self):
        self._ask_data_event.set()

        graph = tf.Graph()
        class_map, _, _ = utils.read_csv()

        with graph.as_default(), tf.Session() as sess:
            model = get_model()

            model_var_names = [v.name for v in tf.global_variables()]

            model_vars = [v for v in tf.global_variables() if v.name in model_var_names]

            # Use a Saver to restore just the variables selected above.
            saver = tf.train.Saver(model_vars, name='model_load_pretrained',
                                   write_version=1)

            checkpoint_path = params.CHECKPOINT_FILE

            saver.restore(sess, checkpoint_path)

            while self.is_running:
                if self._process_buf is None:
                    # Waiting for data to process
                    time.sleep(self._processor_sleep_time)
                    continue

                self._ask_data_event.clear()
                if self._save_path:
                    f_path = os.path.join(self._save_path, 'record_{:.0f}.wav'.format(time.time()))
                    wavfile.write(f_path, self._sample_rate, self._process_buf)
                    logger.info('"{}" saved'.format(f_path))

                logger.info('Start processing')

                # predictions = proc.get_predictions(self._sample_rate, self._process_buf)

                extracted_features = extractor.extract_from_data(self._process_buf / 32768.0, self._sample_rate)

                extracted_features = transpose_features(extracted_features)

                prediction, top_5 = sess.run([model.prediction, model.top_5],
                                             feed_dict={model.features: extracted_features})

                print("\n".join(self.format_top5(class_map, top_5)))
                print("\n\n")
                # formatted = format_predictions(predictions)

                # logger.info('Predictions: {}'.format(formatted))

                # self.events_queue.append((datetime.datetime.now(), formatted))

                logger.info('Stop processing')
                self._process_buf = None
                self._ask_data_event.set()


if __name__ == '__main__':
    server = Daemon(DeviceHiveHandler)
    server.start()
