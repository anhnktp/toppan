EVAL_MIN_CONF = 0.7     # Filter candidate boxes by thresholding the score.
TOP_K_NMS = 400         # How many top scoring bboxes per category are passed to nms
TOP_K_AFTER_NMS = 50    # How many top scoring bboxes per category are left after nms
TOP_K_POST_NMS = 200    # How many top scoring bboxes in total are left after nms for an image
DETECT = True            # if you want a net to perform detection
SEGMENT = False         # if you want a network to perform segmentation
NO_SEG_GT = False

# TRAINING FLAGS
bn_decay = 0.9
learning_rate = 1e-4
TOP_FM = 512            # The number of feature maps in the layers appended to a base network
image_size = 512
#image_size = 512
x4 = True
head = 'nonshared'      #choices=['shared', 'nonshared']
det_kernel = 3          #The size of conv kernel in classification/localization mapping for bboxes
seg_filter_size = 1     # The size of the conv filter used to map feature maps to intermediate representations before segmentation
n_base_channels = 64    # The size of intermediate representations before concatenating and segmenting
resize = 'bilinear'     #choices=['bilinear', 'nearest']
zoomout_prob =  0.5     # To what ratio of images apply zoomout data augmentation
trunk = 'resnet50'      #choices=['resnet50', 'vgg16']      The network you use as a base network (backbone)


