from computation_graph import Node
import math
from itertools import product
from .basic_calculate import Operator


def pad_img(img, padding, P):
    """img: C x H x W -> C x (H+padding) x (W+padding), in zero padding"""
    return P.pad(img, ((0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)


def img2col(img, kernel_size, stride, P):
    """expand image to column formation for convolution"""
    _, h, w = img.shape
    return P.array(
        [img[:, i:i + kernel_size, j:j + kernel_size].flatten() for i in P.arange(0, h - kernel_size + 1, stride) for
         j in P.arange(0, w - kernel_size + 1, stride)])


def kernel2col(kernel):
    """expand conv kernel to column formation for convolution"""
    c = kernel.shape[0]
    return kernel.reshape(c, -1).T


def col2img(result_col, kernel_size, stride, target_shape, P):
    """fold a column back to the original image"""
    _, h, w = target_shape
    c = result_col.shape[0]
    result = P.zeros(shape=target_shape)
    ij = product(P.arange(0, h - kernel_size + 1, stride), P.arange(0, w - kernel_size + 1, stride))
    for iandj, channel in zip(ij, P.arange(c)):
        i, j = iandj[0], iandj[1]
        result[:, i:i + kernel_size, j:j + kernel_size] += result_col[channel].reshape(-1, kernel_size, kernel_size)
    return result


def col2kernel(result, target_shape):
    """fold a column back to convolution kernel"""
    return result.T.reshape(target_shape)


def convolution(kernel, img, kernel_size, stride, padding, result_shape, P):
    """basic convolution operation, use img2col to conv in matmul, then fold the result back"""
    kernel_col = kernel2col(kernel)
    img_col = img2col(pad_img(img, padding, P), kernel_size, stride, P)
    result_col = img_col @ kernel_col
    return result_col.T.reshape(result_shape)


def calculate_output_shape(c, padded_h, padded_w, kernel_size, stride):
    """calculate the output shape of certain convolution"""
    result_h = math.ceil((padded_h - kernel_size + 1) / stride)
    result_w = math.ceil((padded_w - kernel_size + 1) / stride)
    return (c, result_h, result_w)


class Convolve(Operator):
    def __init__(self, *parents, kernel_size=3, padding=0, stride=1, **kwargs):
        super(Operator, self).__init__(*parents, **kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.have_calculate_shape = False
        self.img_shape = ()
        self.padding_shape = ()
        self.result_shape = ()

    def calculate(self):
        # we can only get to know the input shape when calling forward, and we can compute the output shape only after known the input shape, so get to know the input shape by the first forward
        if not self.have_calculate_shape:
            self.img_shape = self.parents[1].value.shape
            self.padding_shape = (
                self.img_shape[0], (self.img_shape[-2] + 2 * self.padding), self.img_shape[-1] + 2 * self.padding)
            self.result_shape = calculate_output_shape(c=self.parents[0].value.shape[0],
                                                       padded_h=self.padding_shape[-2],
                                                       padded_w=self.padding_shape[-1], kernel_size=self.kernel_size,
                                                       stride=self.stride)
            self.have_calculate_shape = True
        self.value = convolution(kernel=self.parents[0].value, img=self.parents[1].value,
                                 kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                 result_shape=self.result_shape, P=self.P)

    def backward(self, parent):
        self.gather_grad()
        grad2col = img2col(img=self.grad, kernel_size=1, stride=1, P=self.P)

        if parent.name == self.parents[0].name:  # grad of kernel, img2col the grad, backward as matrix
            result_grad = img2col(pad_img(self.parents[1].value, self.padding, P=self.P), kernel_size=self.kernel_size,
                                  stride=self.stride, P=self.P).T @ grad2col
            return result_grad.T.reshape(self.parents[0].value.shape)

        else:
            result_grad = grad2col @ kernel2col(self.parents[0].value).T
            result_grad = col2img(result_grad, self.kernel_size, self.stride, self.padding_shape, P=self.P)[:,
                          self.padding:-self.padding, self.padding:-self.padding] \
                if self.padding != 0 \
                else col2img(result_grad,
                             self.kernel_size,
                             self.stride,
                             self.img_shape,
                             P=self.P)  # cut off padding part
            return result_grad


def maxpool(img, kernel_size, padding, stride, result_shape, P):
    padded_img = pad_img(img, padding, P)
    ij = product(P.arange(result_shape[-2]), P.arange(result_shape[-1]))
    result = P.empty(shape=result_shape)
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        result[:, i, j] = padded_img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size].max(
            axis=(-2, -1))
    return result


def find_maxpool_position(img, result, grad, kernel_size, padding, stride, img_shape, result_shape, P):
    """find the corresponding position during the maxpool process, and pass the grad to that position"""
    mask = P.zeros(shape=img_shape)
    ij = product(P.arange(result_shape[-2]), P.arange(result_shape[-1]))
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        # newaxis is to assist board-casting
        index = img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] == result[:, i, j,
                                                                                                    P.newaxis,
                                                                                                    P.newaxis]
        mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] += index * grad[:, i, j,
                                                                                                     P.newaxis,
                                                                                                     P.newaxis]
    return mask[:, padding:-padding, padding:-padding] if padding else mask


class MaxPool(Node):
    def __init__(self, *parents, kernel_size=3, padding=0, stride=1, **kwargs):
        super(MaxPool, self).__init__(*parents, **kwargs)
        assert len(parents) == 1

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.have_calculate_shape = False
        self.img_shape = ()
        self.padding_shape = ()
        self.result_shape = ()

    def calculate(self):
        if not self.have_calculate_shape:
            self.img_shape = self.parents[0].value.shape
            self.padding_shape = (
                self.img_shape[0], (self.img_shape[-2] + 2 * self.padding), self.img_shape[-1] + 2 * self.padding)
            self.result_shape = calculate_output_shape(c=self.img_shape[0],
                                                       padded_h=self.padding_shape[-2],
                                                       padded_w=self.padding_shape[-1], kernel_size=self.kernel_size,
                                                       stride=self.stride)
            self.have_calculate_shape = True
        self.value = maxpool(img=self.parents[0].value, kernel_size=self.kernel_size, stride=self.stride,
                             padding=self.padding,
                             result_shape=self.result_shape, P=self.P)

    def backward(self, parent):
        self.gather_grad()
        grad_in_position = find_maxpool_position(pad_img(self.parents[0].value, self.padding, self.P),
                                                 result=self.value,
                                                 grad=self.grad,
                                                 kernel_size=self.kernel_size, padding=self.padding,
                                                 stride=self.stride, img_shape=self.padding_shape,
                                                 result_shape=self.result_shape,
                                                 P=self.P)
        return grad_in_position


def average_pool(img, kernel_size, padding, stride, result_shape, P):
    padded_img = pad_img(img, padding, P)
    ij = product(P.arange(result_shape[-2]), P.arange(result_shape[-1]))
    result = P.empty(shape=result_shape)
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        result[:, i, j] = padded_img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size].mean(
            axis=(-2, -1))
    return result


def average_pool_grad(grad, kernel_size, padding, stride, img_shape, result_shape, P):
    """calculate the grad passing by the average pool"""
    mask = P.zeros(shape=img_shape)
    scale = 1 / kernel_size ** 2
    ij = product(P.arange(result_shape[-2]), P.arange(result_shape[-1]))
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        # newaxis is to assist board-casting
        mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] += scale * grad[:, i, j,
                                                                                                     P.newaxis,
                                                                                                     P.newaxis]
    return mask[:, padding:-padding, padding:-padding] if padding else mask


class AveragePool(Node):
    def __init__(self, *parents, kernel_size=3, padding=0, stride=1, **kwargs):
        super(AveragePool, self).__init__(*parents, **kwargs)
        assert len(parents) == 1

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.have_calculate_shape = False
        self.img_shape = ()
        self.padding_shape = ()
        self.result_shape = ()

    def calculate(self):
        if not self.have_calculate_shape:
            self.img_shape = self.parents[0].value.shape
            self.padding_shape = (
                self.img_shape[0], (self.img_shape[-2] + 2 * self.padding), self.img_shape[-1] + 2 * self.padding)
            self.result_shape = calculate_output_shape(c=self.img_shape[0],
                                                       padded_h=self.padding_shape[-2],
                                                       padded_w=self.padding_shape[-1], kernel_size=self.kernel_size,
                                                       stride=self.stride)
            self.have_calculate_shape = True
        self.value = average_pool(img=self.parents[0].value, kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding,
                                  result_shape=self.result_shape,
                                  P=self.P)

    def backward(self, parent):
        self.gather_grad()
        grad_in_position = average_pool_grad(grad=self.grad,
                                             kernel_size=self.kernel_size, padding=self.padding,
                                             stride=self.stride, img_shape=self.padding_shape,
                                             result_shape=self.result_shape,
                                             P=self.P)
        return grad_in_position


class GlobalAveragePool(Node):
    def __init__(self, *parents, **kwargs):
        super().__init__(*parents, **kwargs)
        self.kernel_size = -1

    def calculate(self):
        if self.kernel_size < 0:
            self.kernel_size = self.parents[0].value.shape[-1]
        self.value = self.parents[0].value.mean(axis=(-2, -1))

    def backward(self, parent):
        self.gather_grad()
        return self.P.tile(self.grad[..., self.P.newaxis, self.P.newaxis] / (self.kernel_size ** 2),(self.kernel_size,self.kernel_size))


class ReshapeValue(Node):
    def __init__(self, *parents, img_shape, target_shape, **kwargs):
        super(ReshapeValue, self).__init__(*parents, **kwargs)
        assert len(parents) == 1
        self.origin_shape = img_shape
        self.target_shape = target_shape

    def calculate(self):
        self.value = self.parents[0].value.reshape(self.target_shape)

    def backward(self, parent):
        self.gather_grad()
        return self.grad.reshape(self.origin_shape)
