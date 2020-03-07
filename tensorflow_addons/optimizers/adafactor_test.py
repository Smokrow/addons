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
import copy
import numpy as np
from adafactor import AdafactorOptimizer    

class AdafactorTest(tf.test.TestCase):
    def test_setup(self):
        a = AdafactorOptimizer()
        return

    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def test_basic_usage(self):
        x_train = np.random.rand(50,28,28)
        y_train = np.random.randint(0,9,size=(50,10))

        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
        ])

        optimizer = AdafactorOptimizer()
        model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1)
        return

    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def test_get_config(self):
        optimizer = AdafactorOptimizer()
        config = optimizer.get_config()
        self.assertIn("learning_rate",config)
        self.assertIn("multiply_by_parameter_scale",config)
        self.assertIn("decay_rate",config)
        self.assertIn("beta1",config)
        self.assertIn("clipping_threshold",config)
        self.assertIn("factored",config)
        self.assertIn("epsilon1",config)
        self.assertIn("epsilon2",config)


    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def test_change_variables(self):
        val_0 = np.random.random((2,))
        val_1 = np.random.random((2,))

        var_0 = tf.Variable(val_0, dtype=tf.dtypes.float32)
        var_1 = tf.Variable(val_1, dtype=tf.dtypes.float32)

        grad_0 = tf.constant(
            np.random.standard_normal((2,)), dtype=tf.dtypes.float32)
        grad_1 = tf.constant(
            np.random.standard_normal((2,)), dtype=tf.dtypes.float32)

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))
        optimizer = AdafactorOptimizer()

        if tf.executing_eagerly():
            var_0_before = copy.deepcopy(var_0)
            var_1_before = copy.deepcopy(var_1)
            optimizer.apply_gradients(grads_and_vars)
            optimizer.apply_gradients(grads_and_vars)
            self.assertNotEqual(tf.math.reduce_sum(var_0_before),tf.math.reduce_sum(var_0))
            self.assertNotEqual(tf.math.reduce_sum(var_1_before),tf.math.reduce_sum(var_1))

        else:
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self.assertNotEqual(np.sum(np.absolute(val_0)),self.evaluate(tf.math.reduce_sum(tf.math.abs(var_0))))
            self.assertNotEqual(np.sum(np.absolute(val_1)),self.evaluate(tf.math.reduce_sum(tf.math.abs(var_1))))
    

if __name__ == "__main__":
    tf.test.main()
