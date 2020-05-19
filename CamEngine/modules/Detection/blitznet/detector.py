import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import mean_iou
import logging
from .boxer import PriorBoxGrid
from .env import *
from .utils import decode_bboxes, batch_iou

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou
log = logging.getLogger()

class Detector(object):

    def __init__(self, sess, net, loader, config, no_gt=False, CONF_THRESH=0.65, NMS_THRESH=0.6):
        self.sess = sess
        self.net = net
        self.loader = loader
        self.config = config
        self.fm_sizes = self.config['fm_sizes']
        self.no_gt = no_gt
        self.bboxer = PriorBoxGrid(self.config)
        self.score_threshold = CONF_THRESH
        self.nms_threshold = NMS_THRESH
        self.build_detector()


    @staticmethod
    def draw_rectangle(draw, coordinates, color, width=1):
        for i in range(width):
            rect_start = (coordinates[0] - i, coordinates[1] - i)
            rect_end = (coordinates[2] + i, coordinates[3] + i)
            draw.rectangle((rect_start, rect_end), outline=color)


    def restore_from_ckpt(self, weights):
        ckpt_path = weights
        self.sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, ckpt_path)

    def nms(self, localization, confidence, tiling):
        good_bboxes = decode_bboxes(localization, tiling)
        not_crap_mask = tf.reduce_max(confidence[:, 1:], axis=-1) >= self.score_threshold
        good_bboxes = tf.boolean_mask(good_bboxes, not_crap_mask)
        confidence = tf.boolean_mask(confidence, not_crap_mask)

        self.detection_list = []
        self.score_list = []
        for i in range(1, self.loader.num_classes):
            class_mask = tf.greater(confidence[:, i], self.score_threshold)
            class_scores = tf.boolean_mask(confidence[:, i], class_mask)
            class_bboxes = tf.boolean_mask(good_bboxes, class_mask)

            K = tf.minimum(tf.size(class_scores), TOP_K_NMS)
            _, top_k_inds = tf.nn.top_k(class_scores, K)
            top_class_scores = tf.gather(class_scores, top_k_inds)
            top_class_bboxes = tf.gather(class_bboxes, top_k_inds)

            final_inds = tf.image.non_max_suppression(top_class_bboxes,
                                                        top_class_scores,
                                                        max_output_size=TOP_K_AFTER_NMS,
                                                        iou_threshold=self.nms_threshold)

            final_class_bboxes = tf.gather(top_class_bboxes, final_inds)
            final_scores = tf.gather(top_class_scores, final_inds)
            self.detection_list.append(final_class_bboxes)
            self.score_list.append(final_scores)

    def build_detector(self):
        img_size = self.config['image_size']
        self.image_ph = tf.placeholder(shape=[None, None, 3],
                                       dtype=tf.float32, name='img_ph')
        self.seg_ph = tf.placeholder(shape=[None, None], dtype=tf.int32, name='seg_ph')

        img = tf.image.resize_bilinear(tf.expand_dims(self.image_ph, 0),
                                       (img_size, img_size))
        self.net.create_trunk(img)

        if DETECT:
            self.net.create_multibox_head(self.loader.num_classes)
            confidence = tf.nn.softmax(tf.squeeze(self.net.outputs['confidence']))
            location = tf.squeeze(self.net.outputs['location'])
            self.nms(location, confidence, self.bboxer.tiling)

        if SEGMENT:
            self.net.create_segmentation_head(self.loader.num_classes)
            self.segmentation = self.net.outputs['segmentation']
            seg_shape = tf.shape(self.image_ph)[:2]
            self.segmentation = tf.image.resize_bilinear(self.segmentation, seg_shape)

            self.segmentation = tf.cast(tf.argmax(tf.squeeze(self.segmentation), axis=-1), tf.int32)
            self.segmentation = tf.reshape(self.segmentation, seg_shape)
            self.segmentation.set_shape([None, None])

            if not self.no_gt:
                easy_mask = self.seg_ph <= self.loader.num_classes
                predictions = tf.boolean_mask(self.segmentation, easy_mask)
                labels = tf.boolean_mask(self.seg_ph, easy_mask)
                self.mean_iou, self.iou_update = mean_iou(predictions, labels, self.loader.num_classes)
            else:
                self.mean_iou = tf.constant(0)
                self.iou_update = tf.constant(0)

    def process_detection(self, outputs, img, w, h, gt_bboxes, gt_cats, name, draw):
        detection_vec, score_vec = outputs[:2]

        dets, scores, cats = [], [], []
        no_dets = True

        for i in range(self.loader.num_classes-1):
            if score_vec[i].size > 0:
                no_dets = False
                dets.append(detection_vec[i])
                scores.append(score_vec[i])
                cats.append(np.zeros(len(score_vec[i]), dtype='int') + i + 1)

        if not no_dets:
            dets = np.vstack(dets)
            scores = np.concatenate(scores, axis=0)
            cats = np.concatenate(cats, axis=0)

            top_k_inds = np.argsort(scores)[::-1]
            if scores.size > TOP_K_POST_NMS:
                top_k_inds = top_k_inds[0:TOP_K_POST_NMS]
            dets = dets[top_k_inds]
            scores = scores[top_k_inds]
            cats = cats[top_k_inds]

            mask_high = scores >= EVAL_MIN_CONF
            dets = dets[mask_high]
            scores = scores[mask_high]
            cats = cats[mask_high]

            dets[:, :] = dets[:, [1, 0, 3, 2]]
            dets[:, [2, 3]] -= dets[:, [0, 1]]
            dets[:, [0, 2]] *= w
            dets[:, [1, 3]] *= h

        return(dets, scores, cats)

    def process_segmentation(self, outputs, img, seg_gt, name, draw):
        segmentation, iou, _ = outputs[-3:]

        return segmentation, iou

    def get_mean_iou(self):
        iou = self.sess.run(self.mean_iou)
        return iou

    def feed_forward(self, img, seg_gt, w, h, name, gt_bboxes, gt_cats, draw=False):
        feed_dict = {self.image_ph: img}
        net_out = []

        if DETECT:
            net_out.extend([self.detection_list, self.score_list])

        if SEGMENT:
            seg_gt_ = np.zeros(img.shape[:2]) if seg_gt is None else seg_gt
            seg_dict = {self.seg_ph: seg_gt_}
            feed_dict.update(seg_dict)
            net_out.extend([self.segmentation, self.mean_iou, self.iou_update])

        # outputs order with det and seg modes on:
        # detection_vec, score_vec, segmentation, iou, _
        outputs = self.sess.run(net_out, feed_dict=feed_dict)
        results = []

        if DETECT:
            dets, scores, cats = self.process_detection(outputs, img, w, h,
                                                        gt_bboxes, gt_cats,
                                                        name, draw=draw)
            results.extend([dets, scores, cats])

        if SEGMENT:
            segmentation, iou = self.process_segmentation(outputs, img, seg_gt, name, draw)
            results.extend([segmentation, iou])

        return results


def quick_matching(det_boxes, gt_boxes, det_cats, gt_cats):
    iou_mask = batch_iou(det_boxes, gt_boxes) >= 0.5
    det_cats = np.expand_dims(det_cats, axis=1)
    gt_cats = np.expand_dims(gt_cats, axis=0)
    cat_mask = (det_cats == gt_cats)
    matching = np.logical_and(iou_mask, cat_mask)
    return matching
