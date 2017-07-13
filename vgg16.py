import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from scipy.misc import imread, imresize


def vgg16(mat_model_path, input, is_debug=False):
    # load model parameters
    data = loadmat(mat_model_path)

    # read meta info
    meta = data['meta']
    classes = meta['classes']
    class_names = classes[0][0]['description'][0][0][0]
    normalization = meta['normalization']
    average_image = np.squeeze(normalization[0][0]['averageImage'][0][0])

    # read layer info
    layers = data['layers']
    current = input
    network = {}
    with tf.variable_scope('vgg16'):
        for layer in layers[0]:
            name = layer[0]['name'][0][0]
            type = layer[0]['type'][0][0]
            if type == 'conv':
                if name[:2] == 'fc':
                    padding = 'VALID'
                else:
                    padding = 'SAME'
                stride = layer[0]['stride'][0][0][0]
                kernel, bias = layer[0]['weights'][0][0]
                bias = np.squeeze(bias).reshape(-1)
                w = tf.get_variable(
                    name=name + '_w',
                    initializer=tf.constant(kernel)
                )
                b = tf.get_variable(
                    name=name + '_b',
                    initializer=tf.constant(bias)
                )
                current = tf.nn.bias_add(
                    tf.nn.conv2d(
                        input=current,
                        filter=w,
                        strides=(1, stride, stride, 1),
                        padding=padding
                    ), b)
                if is_debug:
                    print name, w.get_shape()
            elif type == 'relu':
                current = tf.nn.relu(current)
                if is_debug:
                    print name, current.get_shape()
            elif type == 'pool':
                stride = layer[0]['stride'][0][0][0]
                pool = layer[0]['pool'][0][0]
                current = tf.nn.max_pool(
                    value=current,
                    ksize=(1, pool[0], pool[1], 1),
                    strides=(1, stride, stride, 1),
                    padding='SAME'
                )
                if is_debug:
                    print name
            elif type == 'softmax':
                current = tf.nn.softmax(tf.reshape(current, [-1, current.get_shape()[-1].value]))
                if is_debug:
                    print name, current.get_shape()

            network[name] = current

    return network, average_image, class_names


if __name__ == '__main__':
    model_path = 'imagenet-vgg-verydeep-16.mat'

    # build graph
    input_img = tf.placeholder(
        dtype=tf.float32,
        shape=(1, 224, 224, 3),
        name='input'
    )
    k = tf.placeholder(
        dtype=tf.int32,
        name='top_k'
    )

    network, average_image, class_names = vgg16(
        mat_model_path=model_path,
        input=input_img,
        is_debug=True
    )

    values, indices = tf.nn.top_k(
        input=network['prob'],
        k=k
    )

    # read sample image
    img = imread('weasel.png', mode='RGB')
    img = imresize(img, [224, 224]) - average_image[-1::-1]

    # testing
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        top_k = 5
        prob, ind = sess.run(
            fetches=[values, indices],
            feed_dict={
                input_img: [img],
                k: top_k
            }
        )
        prob = prob[0]
        ind = ind[0]
        print('\nClassification Result:')
        for i in range(top_k):
            print('\tCategory Name: %s \n\tProbability: %.2f%%\n' % (class_names[ind[i]], prob[i] * 100))

