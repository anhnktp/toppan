import torch
from torch import nn
from .utils import load_state_dict_from_url
from .with_mobilenet import Cpm, InitialStage, RefinementStage

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

# class Cpm(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
#         self.trunk = nn.Sequential(
#             conv_dw_no_bn(out_channels, out_channels),
#             conv_dw_no_bn(out_channels, out_channels),
#             conv_dw_no_bn(out_channels, out_channels)
#         )
#         self.conv = conv(out_channels, out_channels, bn=False)
#
#     def forward(self, x):
#         x = self.align(x)
#         x = self.conv(x + self.trunk(x))
#         return x
#
#
# class InitialStage(nn.Module):
#     def __init__(self, num_channels, num_heatmaps, num_pafs):
#         super().__init__()
#         self.trunk = nn.Sequential(
#             conv(num_channels, num_channels, bn=False),
#             conv(num_channels, num_channels, bn=False),
#             conv(num_channels, num_channels, bn=False)
#         )
#         self.heatmaps = nn.Sequential(
#             conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
#             conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#         self.pafs = nn.Sequential(
#             conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
#             conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#
#     def forward(self, x):
#         trunk_features = self.trunk(x)
#         heatmaps = self.heatmaps(trunk_features)
#         pafs = self.pafs(trunk_features)
#         return [heatmaps, pafs]
#
#
# class RefinementStageBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
#         self.trunk = nn.Sequential(
#             conv(out_channels, out_channels),
#             conv(out_channels, out_channels, dilation=2, padding=2)
#         )
#
#     def forward(self, x):
#         initial_features = self.initial(x)
#         trunk_features = self.trunk(initial_features)
#         return initial_features + trunk_features
#
#
# class RefinementStage(nn.Module):
#     def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
#         super().__init__()
#         self.trunk = nn.Sequential(
#             RefinementStageBlock(in_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels)
#         )
#         self.heatmaps = nn.Sequential(
#             conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
#             conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#         self.pafs = nn.Sequential(
#             conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
#             conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#
#     def forward(self, x):
#         trunk_features = self.trunk(x)
#         heatmaps = self.heatmaps(trunk_features)
#         pafs = self.pafs(trunk_features)
#         return [heatmaps, pafs]


# mobilenet v2

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PoseEstimationWithMobileNet2(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        # begin mobilenet v2
        input_channel = 32
        last_channel = 512
        width_mult = 1.
        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            # [6, 64, 4, 2],
            [6, 96, 3, 1],
            # [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.model = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.model.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.model.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.model.append(conv_1x1_bn(input_channel, last_channel))
        # make it nn.Sequential
        self.model = nn.Sequential(*self.model)
        # print(self.model)
        # end mobilenet v2
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        # print(x.size())
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output


import collections


def _mobilenet2(pretrained, num_refinement_stages=1, **kwargs):
    model = PoseEstimationWithMobileNet2(num_refinement_stages, **kwargs)

    if pretrained:
        model_url = model_urls['mobilenet_v2']
        print('mobilenet_v2')
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format('squeezenet' + version))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=True)

            target_state = model.state_dict()
            new_target_state = collections.OrderedDict()
            for target_key, target_value in target_state.items():
                k = target_key
                if k.find('model') != -1:
                    k = k.replace('model', 'module.model')
                if k in state_dict and state_dict[k].size() == target_state[target_key].size():
                    new_target_state[target_key] = state_dict[k]
                else:
                    new_target_state[target_key] = target_state[target_key]
                    # print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

            model.load_state_dict(new_target_state)

    return model

from torchsummary import summary

if __name__ == "__main__":
    net = _mobilenet2(True)
    summary(net, (3, 368, 368))