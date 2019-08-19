from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave, imread
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import tensorflow.contrib.layers as layers
import tensorflow.keras.backend as K


data_path = 'dataset/'

image_rows = 256
image_cols = 256
n_classes = 3
#smooth = 1.
smooth = 1e-5

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  # some implementations don't square y_pred
  denominator = tf.reduce_sum(y_true + tf.square(y_pred))

  return - numerator / (denominator + tf.keras.backend.epsilon())


def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return - (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def weighted_loss(y_true, y_pred):
    return dice_coef_loss(y_true,y_pred) + K.binary_crossentropy(y_true, y_pred)



def operateWithConstant(input_batch):
    tf_constant = K.constant(np.ones([1, 256,256,1]))
    batch_size = K.shape(input_batch)[0]
    #tiled_constant = K.tile(tf_constant, (batch_size, 256,256,1))
    # Do some operation with tiled_constant and input_batch
    result = tf_constant
    return result



'''
def operateWithConstant(input_batch):
    tf_constant = K.constant(np.ones(256,256,1))
    tf_constant2 = K.constant(np.zeros(256,256,1))
    batch_size = K.shape(input_batch)[0]
    tiled_constant = K.tile(tf_constant, (batch_size, 256,256,1))
    tiled_constant2 = K.tile(tf_constant2, (batch_size, 256,256,1))
    # Do some operation with tiled_constant and input_batch
    result = keras.layers.Add()([tiled_constant, tiled_constant2])
    return result
'''

def BF1(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)
    #one = K.ones(input_shape)
    #constants = [1,2,3]
    #k_constants = K.variable(one)
    #ones = Input(tensor=k_constants)
    ones = Lambda(operateWithConstant)(inputs)
    yb1 = keras.layers.Subtract()([inputs, ones])
    yb2 = MaxPooling2D(pool_size=(3, 3), strides = 1, padding = 'same')(yb1)
    yb3 = keras.layers.Subtract()([yb2, yb1])
    ybext = MaxPooling2D(pool_size = (3,3), strides = 1, padding = 'same')(yb3)
    model = Model(inputs=[inputs], outputs=[yb3, ybext])
    return  model

BF1_model = BF1()
BF1_model.compile(optimizer='adam', loss='binary_crossentropy')


def bf1_error(y_true, y_pred):
    #print(y_true.shape)
    #print(y_true)
    y_true = type(y_true.eval(session=K.get_session()))
    y_pred = type(y_pred.eval(session=K.get_session()))
    ybgt, ybextgt = BF1_model.predict(y_true, verbose=1)
    #print(ybgt)
    ybpd, ybextpd = BF1_model.predict(y_pred, verbose=1)

    num_pc = np.sum(np.multiply(ybpd, ybextgt))
    den_pc = np.sum(ybpd) + 1e-5
    pc = np.divide(num_pc, den_pc)

    num_rc = np.sum(np.multiply(ybgt, ybextpd))
    den_rc = np.sum(ybgt) + 1e-5
    rc = np.divide(num_rc, den_rc)

    BF1c = np.divide(2 * pc*rc, pc + rc + 1e-5)

    return 1 - BF1c


#a= np.zeros([10, 256,256,1])
#b= np.ones([10, 256,256,1])

#print(bf1_error(a,b))





'''

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(true_labels, axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

def mean_iou_loss(y,x):
   return -Mean_IOU(y,x)
'''
n_classes = 3
train_data_path = os.path.join(data_path, 'masked')
annotation_data_path = os.path.join(data_path, 'label')

images = os.listdir(train_data_path)
labels = os.listdir(annotation_data_path)
total = len(images)

imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.float32)
imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.byte)

i = 0
print('-'*30)
print('Creating training images...')
print('-'*30)
#for (image_name, label_name) in zip(images,labels):
for image_name in images:
    img = imread(os.path.join(train_data_path, image_name))
    img_mask = imread(os.path.join(annotation_data_path, image_name), as_gray=True)

    img = np.array([img])
    img_mask = np.array([img_mask])

    imgs[i] = img
    imgs_mask[i] = img_mask

    if i % 100 == 0:
        print('Done: {0}/{1} images'.format(i, total))
    i += 1
print('Loading done.')


imgs_train, imgs_test, imgs_mask_train, imgs_mask_test = train_test_split(imgs, imgs_mask, test_size=0.10, random_state=42)

#np.save('imgs_train.npy', imgs_train)
#np.save('imgs_mask_train.npy', imgs_mask_train)
#np.save('imgs_test.npy', imgs_test)
#np.save('imgs_mask_test.npy', imgs_mask_test)


#del imgs_train
#del imgs_mask_train
#del imgs_test
#del imgs_mask_test


#imgs_train = np.load('imgs_train.npy')
#print(imgs_train)
#imgs_mask_train = np.load('imgs_mask_train.npy')
imgs_mask_train2 = imgs_mask_train

mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = to_categorical(imgs_mask_train, num_classes=n_classes)
#print(imgs_mask_train.shape)


'''
from sklearn.utils.class_weight import compute_class_weight

unique, counts = np.unique(imgs_mask_train2, return_counts=True)


print(unique)
print(counts)

class_weight = dict(zip(unique, compute_class_weight('balanced', unique, imgs_mask_train2.flatten())))
print(class_weight)



from skimage.io import imshow
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt

def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels

        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(y)

    return w

y = generate_random_circles()

wc = {
    0: 1, # background
    1: 5  # objects
}



imgs_mask_train.shape
#w = []
#for i in range(len(imgs_mask_train2)):
#    w.append(unet_weight_map(np.squeeze(imgs_mask_train2[i,:,:]), class_weight))
w = [unet_weight_map(np.squeeze(imgs_mask_train2[i,:,:]), class_weight) for i in range(len(imgs_mask_train2))]
w = np.array(w)
'''


class Network(object):
    def __init__(self):
        super(Network, self).__init__()

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        tf.global_variables_initializer().run() 

    def define_placeholders(self):

        image_placeholder = tf.placeholder(shape=[None, 256, 256, 3], name="img", dtype=tf.float32)
        #y_true_placeholder = tf.placeholder(shape=[None, 256, 256, 3], name='y_true', dtype=tf.float32)
        y_pred_placeholder = tf.placeholder(shape=[None, 256, 256, 3], name='y_pred', dtype=tf.float32)

        return image_placeholder, y_true_placeholder, y_pred_placeholder

    def build_computation_graph(self):

        self.image_placeholder, self.y_true_placeholder, self.y_pred_placeholder = self.define_placeholders()

        self.network = UNet_tf(self.image_placeholder)

        self.output = tf.argmax(self.network, axis = -1)

        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        learning_rate = 1e-5

        #loss = tf.reduce_sum(-self.sy_logprob_n * self.sy_adv_n)

        self.loss = dice_loss(self.output, self.y_pred_placeholder)


        self.train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Define the critic
        # self.critic_prediction = tf.squeeze(build_mlp(
        #                         self.sy_ob_no,
        #                         1,
        #                         "nn_critic",
        #                         n_layers=self.n_layers,
        #                         size=self.size))
        # self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        # # TO-DO: Figure out which one is better
        # self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, self.critic_prediction)
        # self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate_critic).minimize(self.critic_loss)

    def train(self, images, labels):
        loss, _ = self.sess.run([self.loss, self.train], feed_dict={self.image_placeholder: images, self.y_pred_placeholder: labels})
        return loss

    def test(self, images, labels):
        loss, output = self.sess.run([self.loss, self.output], feed_dict={self.image_placeholder: images})
        return loss, output



def UNet_tf(input_placeholder, classes = 3):


    conv1 = layers.convolution2d(input_placeholder, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv1 = layers.convolution2d(conv1, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    pool1 = layers.max_pooling2d(conv1, pool_size = 2, strides = 1)
    pool1 = layers.dropout(pool1)
    pool1 = layers.convolution2d(pool1, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))

    conv2 = layers.convolution2d(pool1, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv2 = layers.convolution2d(conv2, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv2 = conv2 + pool1
    pool2 = layers.max_pooling2d(conv2, pool_size = 2, strides = 1)
    pool2 = layers.dropout(pool2)
    pool2 = layers.convolution2d(pool2, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))

    conv3 = layers.convolution2d(pool2, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv3 = layers.convolution2d(conv3, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv3 = conv3 + pool2
    pool3 = layers.max_pooling2d(conv3, pool_size = 2, strides = 1)
    pool3 = layers.dropout(pool3)
    pool3 = layers.convolution2d(pool3, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))

    conv4 = layers.convolution2d(pool3, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv4 = layers.convolution2d(conv4, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv4 = conv4 + pool3
    pool4 = layers.max_pooling2d(conv4, pool_size = 2, strides = 1)
    pool4 = layers.dropout(pool4)
    pool4 = layers.convolution2d(pool4, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))

    conv5 = layers.convolution2d(pool4, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv5 = layers.convolution2d(conv5, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv5 = conv5 + pool4

    up6 = layers.conv2d_transpose(conv5, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    up6 = tf.concat([up6, conv4])
    up6 = layers.dropout(up6)
    up6 = layers.convolution2d(up6, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv6 = layers.convolution2d(up6, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv6 = layers.convolution2d(conv6, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv6 = conv6 + up6

    up7 = layers.conv2d_transpose(conv6, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    up7 = tf.concat([up7, conv3])
    up7 = layers.dropout(up7)
    up7 = layers.convolution2d(up7, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv7 = layers.convolution2d(up7, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv7 = layers.convolution2d(conv7, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv7 = conv7 + up7

    up8 = layers.conv2d_transpose(conv7, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    up8 = tf.concat([up8, conv2])
    up8 = layers.dropout(up8)
    up8 = layers.convolution2d(up8, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv8 = layers.convolution2d(up8, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv8 = layers.convolution2d(conv8, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv8 = conv8 + up8

    up9 = layers.conv2d_transpose(conv8, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    up9 = tf.concat([up9, conv1])
    up9 = layers.dropout(up9)
    up9 = layers.convolution2d(up9, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv9 = layers.convolution2d(up9, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv9 = layers.convolution2d(conv9, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))
    conv9 = conv9 + up9

    conv10 = layers.convolution2d(conv9, num_outputs=classes, kernel_size=3, stride=1, activation_fn=tf.nn.softmax, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.))

    return conv10


def UNet(input_shape=(256, 256, 3), classes=1):
    inputs = Input(shape=input_shape)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    #conv1 = keras.layers.add([conv1, inputs])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    pool1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = keras.layers.add([conv2, pool1])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    pool2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = keras.layers.add([conv3, pool2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    pool3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = keras.layers.add([conv4, pool3])
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    pool4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)#512

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)#512
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)#512
    conv5 = keras.layers.add([conv5, pool4])

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = Dropout(0.5)(up6)
    up6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = keras.layers.add([conv6, up6])

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    up7 = Dropout(0.5)(up7)
    up7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = keras.layers.add([conv7, up7])

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    up8 = Dropout(0.5)(up8)
    up8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = keras.layers.add([conv8, up8])

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    up9 = Dropout(0.5)(up9)
    up9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = keras.layers.add([conv9, up9])

    conv10 = Conv2D(classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    return model


print(imgs_train[0])


model = Network()
model.init_tf_sess()
model.build_computation_graph()

for epoch in range(100):



#model = UNet(classes=n_classes)
#model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=[dice_coef])
#model.compile(optimizer="adam", loss= dice_coef_loss, metrics=[dice_coef])
#model.compile(optimizer="adam", loss= weighted_loss, metrics=[dice_coef, 'binary_crossentropy', 'categorical_crossentropy'])
#model.compile(optimizer="adam", loss= iou_loss_core, metrics=[dice_loss, dice_coef, 'binary_crossentropy', 'categorical_crossentropy'])

#model.compile(optimizer="adam", loss= mean_iou_loss, metrics=[dice_loss, dice_coef, 'binary_crossentropy', 'categorical_crossentropy'])
#model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=[dice_loss, dice_coef, 'binary_crossentropy', 'categorical_crossentropy'])
#model.compile(optimizer="adam", loss= dice_loss, metrics=[dice_coef, 'binary_crossentropy', 'categorical_crossentropy', bf1_error])


# early_stopping = EarlyStopping(patience=10, verbose=1)
# model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
# reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=False)


# print('-'*30)
# print('Fitting model...')
# print('-'*30)
# model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=100, verbose=2, shuffle=True, #class_weight = w,
#           validation_split=0.1,
#           callbacks=[model_checkpoint, early_stopping, reduce_lr, tensorboard])



from copy import deepcopy
imgs_test_cat = deepcopy(imgs_test)

mean = np.mean(imgs_test)
std = np.std(imgs_test)

imgs_test -= mean
imgs_test /= std

imgs_mask_test = to_categorical(imgs_mask_test, num_classes=3)

# imgs_mask_predict = model.predict(imgs_test, verbose=1)



from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

#def discrete_cmap(N, base_cmap=None):
#    """Create an N-bin discrete colormap from the specified input map"""

#    # Note that if base_cmap is a string or None, you can simply do
#    #    return plt.cm.get_cmap(base_cmap, N)
#    # The following works for string, None, or a colormap instance:

#    base = plt.cm.get_cmap(base_cmap)
#    color_list = base(np.linspace(0, 1, N))
#    cmap_name = base.name + str(N)
#    return base.from_list(cmap_name, color_list, N)

#pred_dir = 'preds'
#if not os.path.exists(pred_dir):
#    os.mkdir(pred_dir)
    
#imgs_test = np.load('imgs_test.npy')



def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

pred_dir = 'preds'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

graphs_dir = 'graphs'
if not os.path.exists(graphs_dir):
    os.mkdir(graphs_dir)
    
# imgs_test = np.load('imgs_test.npy')



plt.rc('text')
imgs_test = imgs_test_cat
#%matplotlib inline
for img_id in range(len(imgs_mask_predict)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2,nrows=2,figsize=(8,8))

    #labels = ['FLORESTA', 'DESMATAMENTO', 'HIDROGRAFIA', 'RESIDUO', 'NUVEM', 'NAO FLORESTA2', 'NAO FLORESTA']
    
    labels = ["Background", "Floresta", "Desmatamento"]

    c = plt.get_cmap('jet', n_classes)

    img_mask_test = np.argmax(imgs_mask_test[img_id], axis = 2)
    img_mask_predict = np.argmax(imgs_mask_predict[img_id], axis = 2)

    im1 = ax1.imshow(imgs_test[img_id].astype(np.uint32))
    ax1.set_title('(a) Imagem Original', x=.5, y=-.15)

    im2 = ax2.imshow(img_mask_test, cmap=c).set_clim(0, n_classes - 1)
    ax2.set_title('(b) Label Original', x=.5, y=-.15)
    
    im3 = ax3.imshow(imgs_test[img_id].astype(np.uint32), alpha=0.5)
    ax3.imshow(np.argmax(imgs_mask_predict[img_id], axis = 2), alpha=0.7, cmap='gray') # OVERLAY
    ax3.set_title('(c) Imagem Original + Label Rede Neural', x=.5, y=-.15)

    im4 = ax4.imshow(img_mask_predict, cmap=c).set_clim(0, n_classes - 1)
    
    ax4.set_title('(d) Label Rede Neural', x=.5, y=-.15)
    
    colors = [c(value) for value in np.arange(0, n_classes)]
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(labels)) ]
    
    plt.draw()



    lgd = f.legend(borderaxespad=0, handles=patches, loc='center')

    bb = lgd.get_bbox_to_anchor().inverse_transformed(ax2.transAxes)
    xOffset = 1.5
    bb.x0 += xOffset
    bb.x1 += xOffset
    lgd.set_bbox_to_anchor(bb, transform = ax2.transAxes)

    plt.tight_layout()

    f.savefig('graphs/graph_{}.png'.format(img_id), format='png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    plt.close(f)
