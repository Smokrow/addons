# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import test_utils
import numpy as np
from adafactor import AdafactorOptimizer    

class AdafactorTest(tf.test.TestCase):
    def test_setup(self):
        a = AdafactorOptimizer()
        return
    def test_basic_usage(self):
        x = np.random.rand(100,28,28)
        y = np.random.rand(100,10)
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
        optimizer = AdafactorOptimizer()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(optimizer.iterations)
        model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy())
        model.fit(x = x, y = y, epochs=10)

        return


if __name__ == "__main__":
    tf.test.main()
