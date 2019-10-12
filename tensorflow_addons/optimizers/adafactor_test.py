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
    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def test_basic_usage(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam()
        optimizer = AdafactorOptimizer()
        #model.compile(optimizer=optimizer,loss="mean_squared_error")
        model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10)
        return
    
    def gradient_tape(self):
        x = np.random.rand(1,2,2)
        y = np.random.rand(1,1)
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(2, 2)),
        tf.keras.layers.Dense(1, activation='softmax')
        ])
        optimizer = AdafactorOptimizer()
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = tf.keras.losses.mean_squared_error(y, predictions)
        print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(gradients)




if __name__ == "__main__":
    tf.test.main()
