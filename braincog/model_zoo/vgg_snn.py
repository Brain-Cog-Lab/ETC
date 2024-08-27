from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule, MultiConvModule
from braincog.datasets import is_dvs_data


@register_model
class SNN7_tiny(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        assert not is_dvs_data(self.dataset), 'SNN7_tiny only support static datasets now'

        self.feature = nn.Sequential(
            BaseConvModule(3, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(16, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, self.num_classes),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


@register_model
class SNN4_tiny(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        assert not is_dvs_data(self.dataset), 'SNN7_tiny only support static datasets now'

        self.feature = nn.Sequential(
            BaseConvModule(3, 28, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2), # 16
            BaseConvModule(28, 56, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(56, 56, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2), # 8
            BaseConvModule(56, 96, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(96, 96, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2), # 4
            BaseConvModule(96, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.num_classes),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


@register_model
class SNN5(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            init_channel = 3
            output_size = 2
        else:
            init_channel = 2
            output_size = 3

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node,
                           n_preact=self.n_preact),
            BaseConvModule(16, 32, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(32, 64, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
        )
        # self.feature = nn.Sequential(
        #     BaseConvModule(init_channel, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
        #     BaseConvModule(16, 64, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
        #     nn.AvgPool2d(2),
        #     BaseConvModule(64, 64, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
        #     nn.AvgPool2d(2),
        #     BaseConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
        #     nn.AvgPool2d(2),
        #     BaseConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
        #     nn.AvgPool2d(2),
        # )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * output_size * output_size, self.num_classes * 10),
            VotingLayer(10),
        )
        print(self.fc)

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


@register_model
class VGG_SNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            init_channel = 3
            out_size = 2 ** 2
        else:
            init_channel = 2
            out_size = 3 ** 2

        self.feature = nn.Sequential(
            BaseConvModule(init_channel * self.init_channel_mul, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * out_size, self.num_classes),
        )

    def forward(self, inputs):

        if not self.online_training:
            inputs = self.encoder(inputs).contiguous()
            self.reset()

            if self.layer_by_layer:
                x = self.feature(inputs)
                x = self.fc(x)
                if self.tet_loss:
                    x = rearrange(x, '(t b) c -> t b c', t=self.step)
                else:
                    x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
                return x

            else:
                outputs = []
                for t in range(self.step):
                    x = inputs[t]
                    x = self.feature(x)
                    x = self.fc(x)
                    outputs.append(x)
                # self.outputs = outputs

                if self.tet_loss:
                    return torch.stack(outputs, dim=0)
                else:
                    return sum(outputs) / len(outputs)
                    # return sum(outputs[1:]) / (len(outputs) - 1)
        else:
            x = inputs
            x = self.feature(x)
            x = self.fc(x)
            return x


@register_model
class VGG7(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            init_channel = 3
            out_shape = 2
        else:
            init_channel = 2
            out_shape = 3

        self.feature = nn.Sequential(
            BaseConvModule(init_channel * self.init_channel_mul, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * out_shape ** 2, self.num_classes),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs).contiguous()
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            if self.tet_loss:
                x = rearrange(x, '(t b) c -> t b c', t=self.step)
            else:
                x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)
            # self.outputs = outputs

            if self.tet_loss:
                return torch.stack(outputs, dim=0)
            else:
                return sum(outputs) / len(outputs)
                # return sum(outputs[1:]) / (len(outputs) - 1)


@register_model
class VGG7_half(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            init_channel = 3
            out_shape = 2
        else:
            init_channel = 2
            out_shape = 3

        self.feature = nn.Sequential(
            BaseConvModule(init_channel * self.init_channel_mul, 32, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(32, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * out_shape ** 2, self.num_classes),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs).contiguous()
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            if self.tet_loss:
                x = rearrange(x, '(t b) c -> t b c', t=self.step)
            else:
                x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)
            # self.outputs = outputs

            if self.tet_loss:
                return torch.stack(outputs, dim=0)
            else:
                return sum(outputs) / len(outputs)
                # return sum(outputs[1:]) / (len(outputs) - 1)

@register_model
class Vgglike(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if is_dvs_data(self.dataset):
            data_channel = 2
        else:
            data_channel = 3

        self.feature = nn.Sequential(
            BaseConvModule(data_channel * self.init_channel_mul, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 576, bias=True),
            self.node(),
            nn.Linear(576, self.num_classes)
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs).contiguous()
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            if self.tet_loss:
                x = rearrange(x, '(t b) c -> t b c', t=self.step)
            else:
                x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)
            # self.outputs = outputs

            if self.tet_loss:
                return torch.stack(outputs, dim=0)
            else:
                return sum(outputs) / len(outputs)
                # return sum(outputs[1:]) / (len(outputs) - 1)


@register_model
class Vgg16CifarS(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if is_dvs_data(self.dataset):
            data_channel = 2
        else:
            data_channel = 3

        self.feature = nn.Sequential(
            BaseConvModule(data_channel * self.init_channel_mul, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
            nn.MaxPool2d(2, 2)
        )
        # self.feature = nn.Sequential(
        #     BaseConvModule(data_channel * self.init_channel_mul, 64, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1),
        #                    node=nn.Identity, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1),
        #                    node=nn.Identity, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=nn.Identity, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=nn.Identity, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2)
        # )
        # self.feature = nn.Sequential(
        #     BaseConvModule(data_channel * self.init_channel_mul, 64, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(64, 128, kernel_size=(5, 5), padding=(2, 2),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     # BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1),
        #     #                node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(128, 256, kernel_size=(5, 5), padding=(2, 2),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     # BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
        #     #                node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(256, 512, kernel_size=(5, 5), padding=(2, 2),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     # BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #     #                node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     BaseConvModule(512, 512, kernel_size=(5, 5), padding=(2, 2),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     # BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #     #                node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                    node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2)
        # )
        # self.feature = nn.Sequential(
        #     MultiConvModule(data_channel * self.init_channel_mul, 64, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(64, 64, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     MultiConvModule(64, 128, kernel_size=(5, 5), padding=(2, 2),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     MultiConvModule(128, 256, kernel_size=(5, 5), padding=(2, 2),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     MultiConvModule(256, 512, kernel_size=(5, 5), padding=(2, 2),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2),
        #     MultiConvModule(512, 512, kernel_size=(5, 5), padding=(2, 2),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     MultiConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
        #                     node=self.node, n_preact=self.n_preact, conv_fun=self.conv_fun),
        #     nn.MaxPool2d(2, 2)
        # )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.num_classes, bias=True),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs).contiguous()
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            if self.tet_loss:
                x = rearrange(x, '(t b) c -> t b c', t=self.step)
            else:
                x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)
            # self.outputs = outputs

            if self.tet_loss:
                return torch.stack(outputs, dim=0)
            else:
                return sum(outputs) / len(outputs)
                # return sum(outputs[1:]) / (len(outputs) - 1)
