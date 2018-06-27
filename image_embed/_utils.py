# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def cut_model(model_raw, depth):
    from keras import Model
    layer_names = [x.name for x in model_raw.layers if x.count_params() > 0]
    if (depth > len(layer_names)) or (depth < -1 * len(layer_names)):
        msg = "Selected model has only {0:d} layers"
        raise IndexError(msg.format(len(layer_names)))

    model = Model(inputs=model_raw.input,
                  outputs=model_raw.get_layer(layer_names[depth]).output)

    return model


def load_model(model_name, depth):

    if model_name == "xception":
        res = load_xception(depth)
    elif model_name == "vgg16":
        res = load_vgg16(depth)
    elif model_name == "vgg19":
        res = load_vgg19(depth)
    elif model_name == "resnet50":
        res = load_resnet50(depth)
    elif model_name == "inception_v3":
        res = load_inception_v3(depth)
    elif model_name == "inception_resnet_v2":
        res = load_inception_resnet_v2(depth)
    elif model_name == "mobilenet":
        res = load_mobilenet(depth)
    elif model_name == "densenet121":
        res = load_densenet121(depth)
    elif model_name == "densenet169":
        res = load_densenet169(depth)
    elif model_name == "densenet201":
        res = load_densenet201(depth)
    else:
        raise ValueError("model name not found")

    return res


def load_xception(depth=-2):
    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input

    model_raw = Xception()
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (299, 299)


def load_vgg16(depth=-2):
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input

    model_raw = VGG16()
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (224, 224)


def load_vgg19(depth=-2):
    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input

    model_raw = VGG19()
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (224, 224)


def load_resnet50(depth=-2):
    from keras.applications.resnet50 import ResNet50
    from keras.applications.resnet50 import preprocess_input

    model_raw = ResNet50(weights='imagenet')
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (224, 224)


def load_inception_v3(depth=-2):
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input

    model_raw = InceptionV3(weights='imagenet')
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (299, 299)


def load_inception_resnet_v2(depth=-2):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.inception_resnet_v2 import preprocess_input

    model_raw = InceptionResNetV2(weights='imagenet')
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (299, 299)


def load_mobilenet(depth=-2):
    from keras.applications.mobilenet import MobileNet
    from keras.applications.mobilenet import preprocess_input

    model_raw = MobileNet()
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (224, 224)


def load_densenet121(depth=-2):
    from keras.applications.densenet import DenseNet121
    from keras.applications.densenet import preprocess_input

    model_raw = DenseNet121()
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (224, 224)


def load_densenet169(depth=-2):
    from keras.applications.densenet import DenseNet169
    from keras.applications.densenet import preprocess_input

    model_raw = DenseNet169()
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (224, 224)


def load_densenet201(depth=-2):
    from keras.applications.densenet import DenseNet201
    from keras.applications.densenet import preprocess_input

    model_raw = DenseNet201()
    model = cut_model(model_raw, depth)

    return model, preprocess_input, (224, 224)
