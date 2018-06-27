# -*- coding: utf-8 -*-
"""Embedding images using neural networks

This module provides a high-level interface for projecting
images into meaningful strings of numbers. The embedding are
made by looking at the interior projections from popular
neural networks. The module serves as a wrapper around the
popular keras package, seamlessly handling the image i/o,
preprocessing steps, and extraction of interior layers. Helper
functions for image clustering and similarity detection are
also included.

The api package describes the main entry points for user written
python code. There is also a command-line tool for producing the
embeddings directly as a csv file.

Example:

    from image_embed import ImageEmbedder
    ie = ImageEmbedder()
    ie.load_model('vgg19', depth=-2)
    img_paths = ["file1.jpg", "file2.jpg"]
    z = ie.process_images(img_paths)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ._utils import load_model


MODEL_NAMES = ['xception', 'vgg16', 'vgg19', 'resnet50', 'inception_v3',
               'inception_resnet_v2', 'mobilenet', 'densenet121',
               'densenet169', 'densenet201']
"""Names of models supported by the ImageEmbedder class
"""


class ImageEmbedder:
    """Object for computing embeddings

    Attributes:
        nn: The keras neural network object.
        pp: Object to pre-process images; set to None if no such processing
            needed.
        ts: Target size (height and width) of the input images.
    """
    nn = None
    pp = None
    ts = None

    def set_model(self, model, preprocessor, ts):
        """Manually set model parameters

        For advanced users, this method allows for manually specifying
        the model and preprocessor attributes of a class instance.

        Args:
            model: The keras model to fit to the data.
            preprocessor: An optional function to apply to each image prior
                to fitting the model.
            ts: A tuple giving the height and width required by images input
                into the model.
        """
        self.nn = model
        self.pp = pp
        self.ts = model.layers[0].input_shape[1:3]

    def load_model(self, model_name="mobilenet", depth=-2):
        """Load preconstructed image processing model

        This method loads a keras model, with an option to output the
        intermediate results in a non-output layer. The preprocessing
        functions and required input sizes are also automatically
        set by this method.

        Args:
            model_name: String describing the model name. See the
                object MODEL_NAMES of allowed model names.
            depth: An integer giving the desired depth of the model.
                Specifically, it functions like list slicing; it will return
                layers zero through depth-1. Set to a negative number to
                index from the top. For example, depth equal to -1 return the
                original output. The default is set to -2, the penultimate
                layer, which is the most commonly used for image similarity
                metrics. Note that only layers with non-zero parameters are
                included in the count.
        """
        self.nn, self.pp, self.ts = load_model(model_name, depth)

    def show_model(self):
        """Print a loaded model's architecture
        """
        self.nn.summary()

    def process_images(self, img_paths, pooling='flat', verbose=1):
        """Process collection of image files

        This method takes a list of image paths, reads in each image,
        applies the specific preprocessor and model, and returns a numpy
        array containing the resulting embedding. Currently the image paths
        must be a list, not a generator; this is done for efficiency in
        being able to pre-populate the in-memory array of images. Future
        version will allow for out-of-memory processing over image batches.

        Args:
            img_paths: A list of file paths describing the location of the
                images. Can also be a single string, which will be converted
                into a list of length 1.
            pooling: If a layer is selected that returns a 4D tensor, this
                selects the type of pooling that is selected. If 'none', the
                raw results are returned; 'avg' uses global average pooling;
                'max' uses global max pooling; 'flat' returns a flattened
                version of the tensor. When using pooling, the results will
                be flattened to return a 2-dimensional output. Other options
                will default to flattening the results.
            verbose: An integer specifying whether the fitting routine should
                verbosely show a progress bar. Set to 0 to turn off the bar
                and keep it equal to 1 (the default) to produce a progress
                bar.

        Returns:
            A numpy array of the embeddings for each image.
        """
        import keras.preprocessing

        if self.nn is None or self.ts is None:
            raise ValueError("You must initialize a model before processing.")

        if isinstance(img_paths, str):
            img_paths = [img_paths]

        x = np.zeros([len(img_paths)] + list(self.ts) + [3])
        for i, val in enumerate(img_paths):
            img = keras.preprocessing.image.load_img(img_paths[i],
                                                     target_size=self.ts)
            x[i, :, :, :] = keras.preprocessing.image.img_to_array(img)

        return self.process_array(x, pooling, verbose)

    def process_array(self, x, pooling='flat', verbose=1):
        """Process images stored as a numpy array

        This method takes a raw numpy array and applies the model to the
        data. The method is called internally by the process_images method;
        only one of the two needs to be called by the user.

        Args:
            x: A numpy array of images, which must match to dimensions of the
                model loaded into the ImageEmbedder instance.
            pooling: If a layer is selected that returns a 4D tensor, this
                selects the type of pooling that is selected. If 'none', the
                raw results are returned; 'avg' uses global average pooling;
                'max' uses global max pooling; 'flat' returns a flattened
                version of the tensor. When using pooling, the results will
                be flattened to return a 2-dimensional output. Other options
                will default to flattening the results.
            verbose: An integer specifying whether the fitting routine should
                verbosely show a progress bar. Set to 0 to turn off the bar
                and keep it equal to 1 (the default) to produce a progress
                bar.

        Returns:
            A numpy array of the embeddings for each image.
        """
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)

        if self.pp is not None:
            x = self.pp(x)

        emb = self.nn.predict(x, verbose=verbose)
        if pooling == "none" or len(emb.shape) < 3:
            pass
        elif pooling == "max":
            emb = np.max(emb, 3)
            emb = emb.reshape((emb.shape[0], np.prod(emb.shape[1:])))
        elif pooling == "avg":
            emb = np.mean(emb, 3)
            emb = emb.reshape((emb.shape[0], np.prod(emb.shape[1:])))
        else:
            emb = emb.reshape((emb.shape[0], np.prod(emb.shape[1:])))

        return emb
