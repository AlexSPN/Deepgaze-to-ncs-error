#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
import os.path

class CnnHeadPoseEstimator:
    def __init__(self):

        self._sess = tf.Session()

    def _allocate_yaw_variables(self):
        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_yaw_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3), name="input")

        #Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hy_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hy_conv1_biases = tf.Variable(tf.zeros([64]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hy_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hy_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        #here 5*5 is the size of the image after pool reduction (divide by half 3 times)
        self.hy_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hy_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Output layer
        self.hy_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hy_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))

        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])
            conv1 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hy_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv1_biases))
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            conv2 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hy_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv2_biases))
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            conv3 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hy_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv3_biases))
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            dense1 = tf.reshape(norm3, [-1, self.hy_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            dense1 = tf.nn.tanh(tf.matmul(dense1, self.hy_dense1_weights) + self.hy_dense1_biases)
            out = tf.nn.tanh(tf.matmul(dense1, self.hy_out_weights) + self.hy_out_biases, name="output")
            return out

        # Get the result from the model
        self.cnn_yaw_output = model(self.tf_yaw_input_vector)


    def load_yaw_variables(self, YawFilePath):
        #Allocate the variables in memory
        self._allocate_yaw_variables()

        if(os.path.isfile(YawFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_yaw_variables): the yaw file path is incorrect.')

        tf.train.Saver(({"conv1_yaw_w": self.hy_conv1_weights, "conv1_yaw_b": self.hy_conv1_biases,
                         "conv2_yaw_w": self.hy_conv2_weights, "conv2_yaw_b": self.hy_conv2_biases,
                         "conv3_yaw_w": self.hy_conv3_weights, "conv3_yaw_b": self.hy_conv3_biases,
                         "dense1_yaw_w": self.hy_dense1_weights, "dense1_yaw_b": self.hy_dense1_biases,
                         "out_yaw_w": self.hy_out_weights, "out_yaw_b": self.hy_out_biases
                        })).restore(self._sess, YawFilePath)


        writer = tf.summary.FileWriter('tanh_tensorboard_logs', self._sess.graph)
        writer.close()


        saver = tf.train.Saver()
        saver.save(self._sess,"tanh_ncs/tf_model")


if __name__ == "__main__":
    estimator = CnnHeadPoseEstimator()
    estimator.load_yaw_variables("./data/yaw/cnn_cccdd_30k.tf")


