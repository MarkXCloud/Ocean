from computation_graph import Node
import numpy as np
import cupy as cp
import math
from itertools import product
from .basic_calculate import Operator


def pad_img(img, padding):
    """img: C x H x W -> C x (H+padding) x (W+padding), in zero padding"""
    return np.pad(img, ((0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)


def pad_img_gpu(img, padding):
    return cp.pad(img, ((0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)


def img2col(img, kernel_size, stride):
    """expand image to column formation for convolution"""
    _, h, w = img.shape
    return np.array(
        [img[:, i:i + kernel_size, j:j + kernel_size].flatten() for i in range(0, h - kernel_size + 1, stride) for
         j in range(0, w - kernel_size + 1, stride)])


def img2col_gpu(img, kernel_size, stride):
    _, h, w = img.shape
    return cp.array(
        [img[:, i:i + kernel_size, j:j + kernel_size].flatten() for i in cp.arange(0, h - kernel_size + 1, stride)
         for j in cp.arange(0, w - kernel_size + 1, stride)])


def kernel2col(kernel):
    """expand conv kernel to column formation for convolution"""
    c = kernel.shape[0]
    return kernel.reshape(c, -1).T


def col2img(result_col, kernel_size, stride, target_shape):
    """fold a column back to the original image"""
    c, h, w = target_shape
    result = np.zeros(shape=target_shape)
    ij = product(range(0, h - kernel_size + 1, stride), range(0, w - kernel_size + 1, stride))
    for iandj, channel in zip(ij, range(c)):
        i, j = iandj[0], iandj[1],
        result[:, i:i + kernel_size, j:j + kernel_size] += result_col[channel].reshape(-1, kernel_size, kernel_size)
    return result


def col2img_gpu(result_col, kernel_size, stride, target_shape):
    c, h, w = target_shape
    result = cp.zeros(shape=target_shape)
    ij = product(cp.arange(0, h - kernel_size + 1, stride), cp.arange(0, w - kernel_size + 1, stride))
    for iandj, channel in zip(ij, cp.arange(c)):
        i, j = iandj[0], iandj[1],
        result[:, i:i + kernel_size, j:j + kernel_size] = cp.add(result[:, i:i + kernel_size, j:j + kernel_size],
                                                                 result_col[channel].reshape(-1, kernel_size,
                                                                                             kernel_size))
    return result


def col2kernel(result, target_shape):
    """fold a column back to convolution kernel"""
    return result.T.reshape(target_shape)


def convolution(kernel, img, kernel_size, stride, padding, result_shape):
    """basic convolution operation, use img2col to conv in matmul, then fold the result back"""
    kernel_col = kernel2col(kernel)
    img_col = img2col(pad_img(img, padding), kernel_size, stride)
    result_col = img_col @ kernel_col
    return result_col.T.reshape(result_shape)


def convolution_gpu(kernel, img, kernel_size, stride, padding, result_shape):
    kernel_col = kernel2col(kernel)
    img_col = img2col_gpu(pad_img(img, padding), kernel_size, stride)
    result_col = cp.matmul(img_col, kernel_col)
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
        if self.graph.cuda_device == 'cpu':
            self.value = convolution(kernel=self.parents[0].value, img=self.parents[1].value,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     result_shape=self.result_shape)
        else:
            self.value = convolution_gpu(kernel=self.parents[0].value, img=self.parents[1].value,
                                         kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                         result_shape=self.result_shape)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            grad2col = img2col(img=self.grad, kernel_size=1, stride=1)
            if parent.name == self.parents[0].name:  # grad of kernel, img2col the grad, backward as matrix
                result_grad = img2col(pad_img(self.parents[1].value, self.padding), kernel_size=self.kernel_size,
                                      stride=self.stride).T @ grad2col
                return result_grad.T.reshape(self.parents[0].value.shape)

            else:
                result_grad = grad2col @ kernel2col(self.parents[0].value).T
                result_grad = col2img(result_grad, self.kernel_size, self.stride, self.padding_shape)[:,
                              self.padding:-self.padding, self.padding:-self.padding] \
                    if self.padding != 0 \
                    else col2img(result_grad,
                                 self.kernel_size,
                                 self.stride,
                                 self.img_shape)  # cut off padding part
                return result_grad
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
            grad2col = img2col(img=self.grad, kernel_size=1, stride=1)
            if parent.name == self.parents[0].name:
                result_grad = cp.matmul(
                    img2col_gpu(pad_img_gpu(self.parents[1].value, self.padding), kernel_size=self.kernel_size,
                                stride=self.stride).T, grad2col)
                return result_grad.T.reshape(self.parents[0].value.shape)
            else:
                result_grad = cp.matmul(grad2col, kernel2col(self.parents[0].value).T)
                result_grad = col2img_gpu(result_grad, self.kernel_size, self.stride, self.padding_shape)[:,
                              self.padding:-self.padding, self.padding:-self.padding] \
                    if self.padding != 0 \
                    else col2img_gpu(result_grad,
                                     self.kernel_size,
                                     self.stride,
                                     self.padding_shape)  # cut off padding part
                return result_grad


def maxpool(img, kernel_size, padding, stride, result_shape):
    padded_img = pad_img(img, padding)
    ij = product(range(result_shape[-2]), range(result_shape[-1]))
    result = np.empty(shape=result_shape)
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        result[:, i, j] = padded_img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size].max(
            axis=(-2, -1))
    return result


def maxpool_gpu(img, kernel_size, padding, stride, result_shape):
    padded_img = pad_img_gpu(img, padding)
    ij = product(cp.arange(result_shape[-2]), cp.arange(result_shape[-1]))
    result = cp.empty(shape=result_shape)
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        result[:, i, j] = padded_img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size].max(
            axis=(-2, -1))
    return result


def find_maxpool_position(img, result, grad, kernel_size, padding, stride, img_shape, result_shape):
    """find the corresponding position during the maxpool process, and pass the grad to that position"""
    mask = np.zeros(shape=img_shape)
    ij = product(range(result_shape[-2]), range(result_shape[-1]))
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        # newaxis is to assist board-casting
        index = img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] == result[:, i, j,
                                                                                                    np.newaxis,
                                                                                                    np.newaxis]
        mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] += index * grad[:, i, j,
                                                                                                     np.newaxis,
                                                                                                     np.newaxis]
    return mask[:, padding:-padding, padding:-padding] if padding else mask


def find_maxpool_position_gpu(img, result, grad, kernel_size, padding, stride, img_shape, result_shape):
    mask = cp.zeros(shape=img_shape)
    ij = product(cp.arange(result_shape[-2]), cp.arange(result_shape[-1]))
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        index = cp.equal(img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size],
                         result[:, i, j, cp.newaxis, cp.newaxis])
        mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] = cp.add(
            mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size],
            index * grad[:, i, j, cp.newaxis, cp.newaxis])
    return mask[:, padding:-padding, padding:-padding] if padding else mask


class MaxPool(Node):
    def __init__(self, *parents,  kernel_size=3, padding=0, stride=1, **kwargs):
        super(MaxPool, self).__init__(*parents, **kwargs)
        assert len(parents) == 1

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.have_calculate_shape=False
        self.img_shape = ()
        self.padding_shape = ()
        self.result_shape =()

    def calculate(self):
        if not self.have_calculate_shape:
            self.img_shape = self.parents[0].value.shape
            self.padding_shape = (
                self.img_shape[0], (self.img_shape[-2] + 2 * self.padding), self.img_shape[-1] + 2 * self.padding)
            self.result_shape = calculate_output_shape(c=self.img_shape[0],
                                                       padded_h=self.padding_shape[-2],
                                                       padded_w=self.padding_shape[-1], kernel_size=self.kernel_size,
                                                       stride=self.stride)
            self.have_calculate_shape=True
        if self.graph.cuda_device == 'cpu':
            self.value = maxpool(img=self.parents[0].value, kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding,
                                 result_shape=self.result_shape)
        else:
            self.value = maxpool_gpu(img=self.parents[0].value, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding,
                                     result_shape=self.result_shape)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            grad_in_position = find_maxpool_position(pad_img(self.parents[0].value, self.padding), result=self.value,
                                                     grad=self.grad,
                                                     kernel_size=self.kernel_size, padding=self.padding,
                                                     stride=self.stride, img_shape=self.padding_shape,
                                                     result_shape=self.result_shape)
            return grad_in_position
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
            grad_in_position = find_maxpool_position_gpu(self.parents[0].value, result=self.value, grad=self.grad,
                                                         kernel_size=self.kernel_size, padding=self.padding,
                                                         stride=self.stride, img_shape=self.padding_shape,
                                                         result_shape=self.result_shape)
            return grad_in_position


def average_pool(img, kernel_size, padding, stride, result_shape):
    padded_img = pad_img(img, padding)
    ij = product(range(result_shape[-2]), range(result_shape[-1]))
    result = np.empty(shape=result_shape)
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        result[:, i, j] = padded_img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size].mean(
            axis=(-2, -1))
    return result


def average_pool_gpu(img, kernel_size, padding, stride, result_shape):
    padded_img = pad_img_gpu(img, padding)
    ij = product(range(result_shape[-2]), range(result_shape[-1]))
    result = cp.empty(shape=result_shape)
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        result[:, i, j] = padded_img[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size].mean(
            axis=(-2, -1))
    return result


def average_pool_grad(grad, kernel_size, padding, stride, img_shape, result_shape):
    """calculate the grad passing by the average pool"""
    mask = np.zeros(shape=img_shape)
    scale = 1 / kernel_size ** 2
    ij = product(range(result_shape[-2]), range(result_shape[-1]))
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        # newaxis is to assist board-casting
        mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] += scale * grad[:, i, j,
                                                                                                     np.newaxis,
                                                                                                     np.newaxis]
    return mask[:, padding:-padding, padding:-padding] if padding else mask


def average_pool_grad_gpu(grad, kernel_size, padding, stride, img_shape, result_shape):
    """calculate the grad passing by the average pool"""
    mask = cp.zeros(shape=img_shape)
    scale = 1 / kernel_size ** 2
    ij = product(range(result_shape[-2]), range(result_shape[-1]))
    for iandj in ij:
        i, j = iandj[0], iandj[1]
        # newaxis is to assist board-casting
        mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size] = cp.add(
            mask[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size],
            cp.multiply(scale, grad[:, i, j, cp.newaxis, cp.newaxis]))
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
        if self.graph.cuda_device == 'cpu':
            self.value = average_pool(img=self.parents[0].value, kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding,
                                      result_shape=self.result_shape)
        else:
            self.value = average_pool_gpu(img=self.parents[0].value, kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding,
                                          result_shape=self.result_shape)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            grad_in_position = average_pool_grad(grad=self.grad,
                                                 kernel_size=self.kernel_size, padding=self.padding,
                                                 stride=self.stride, img_shape=self.padding_shape,
                                                 result_shape=self.result_shape)
            return grad_in_position
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]),
                                   axis=0)  # gather grad
            grad_in_position = average_pool_grad_gpu(grad=self.grad,
                                                     kernel_size=self.kernel_size, padding=self.padding,
                                                     stride=self.stride, img_shape=self.padding_shape,
                                                     result_shape=self.result_shape)
            return grad_in_position


class ReshapeValue(Node):
    def __init__(self, *parents, img_shape, target_shape, **kwargs):
        super(ReshapeValue, self).__init__(*parents, **kwargs)
        assert len(parents) == 1
        self.origin_shape = img_shape
        self.target_shape = target_shape

    def calculate(self):
        self.value = self.parents[0].value.reshape(self.target_shape)

    def backward(self, parent):
        if self.graph.cuda_device == 'cpu':
            if self.judge_nan(self.grad):
                self.grad = np.sum([child.backward(self) for child in self.children], axis=0)  # gather grad
            return self.grad.reshape(self.origin_shape)
        else:
            if self.judge_nan(self.grad):
                self.grad = cp.sum(cp.asarray([child.backward(self) for child in self.children]), axis=0)  # gather grad
            return self.grad.reshape(self.origin_shape)
