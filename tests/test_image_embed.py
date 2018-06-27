# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest
import os
import numpy as np

from image_embed import ImageEmbedder, MODEL_NAMES


class Test_image_input(unittest.TestCase):

    def test_img_similar(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png", "eiffel2.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        sim = np.mean(np.abs(embed[0, :] - embed[1, :]))

        self.assertLess(sim, 3)

    def test_img_png_jpg(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png", "eiffel1.jpg"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        sim = np.mean(np.abs(embed[0, :] - embed[1, :]))

        self.assertLess(sim, 0.0001)

    def test_img_alpha(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png",
                                              "eiffel1-alpha.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        sim = np.mean(np.abs(embed[0, :] - embed[1, :]))

        self.assertLess(sim, 0.0001)

    def test_img_small(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png",
                                              "eiffel1-small.jpg",
                                              "eiffel1-small.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        sim1 = np.mean(np.abs(embed[0, :] - embed[1, :]))
        sim2 = np.mean(np.abs(embed[0, :] - embed[2, :]))
        sim = (sim1 + sim2) / 2

        self.assertLess(sim, 2)

    def test_img_bw(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png",
                                              "eiffel1-bw.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        sim = np.mean(np.abs(embed[0, :] - embed[1, :]))

        self.assertLess(sim, 3)

    def test_img_bw_alpha(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1-bw.png",
                                              "eiffel1-bw-alpha.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        sim = np.mean(np.abs(embed[0, :] - embed[1, :]))

        self.assertLess(sim, 0.0001)

    def test_img_diff(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png",
                                              "dog.jpg"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        sim = np.mean(np.abs(embed[0, :] - embed[1, :]))

        self.assertGreater(sim, 2)

    def test_img_one(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=-2)
        embed = ie.process_images(imgs, pooling="flat")
        embed_one = ie.process_images(imgs[0], pooling="flat")
        sim = np.mean(np.abs(embed[0, :] - embed_one[0, :]))

        self.assertLess(sim, 0.000001)


class Test_model(unittest.TestCase):

    # def test_each_model_loads(self):
    #     ie = ImageEmbedder()
    #     for model_name in MODEL_NAMES:
    #         ie.load_model(model_name=model_name, depth=1)

    def test_pooling(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=2)
        embed_none = ie.process_images(imgs, pooling="none")
        embed_max = ie.process_images(imgs, pooling="max")
        embed_avg = ie.process_images(imgs, pooling="avg")
        embed_flat = ie.process_images(imgs, pooling="flat")

        self.assertEqual(embed_none.shape, (1, 112, 112, 32))
        self.assertEqual(embed_max.shape, (1, 112 * 112))
        self.assertEqual(embed_avg.shape, (1, 112 * 112))
        self.assertEqual(embed_flat.shape, (1, 112 * 112 * 32))
        self.assertGreater(np.mean(embed_max), np.mean(embed_avg))

    def test_depth_positive(self):
        bp = os.path.join(os.path.dirname(__file__), 'test-images')
        imgs = [os.path.join(bp, x) for x in ["eiffel1.png"]]

        ie = ImageEmbedder()
        ie.load_model(model_name="mobilenet", depth=10)
        embed_pos = ie.process_images(imgs)
        ie.load_model(model_name="mobilenet", depth=-45)
        embed_neg = ie.process_images(imgs)

        sim = np.sum(np.abs(embed_pos[0, :] - embed_neg[0, :]))

        self.assertLess(sim, 0.000001)


