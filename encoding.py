import torch
from torch.nn.functional import one_hot
import symlog

"""
Two hot encoding

Soft categorical encoding scheme for integer and real numbers

Numbers are represented on a line between min -- max
The line is demarcated with C classes, and the two nearest
classes to the number being represented hold values which sum to one
the bucket closest to the number has the larger value

ie: to represent the numbers between 0 .. 1 with 2 Classes...

0.0 -> [1.0, 0.0]
0.1 -> [0.9, 0.1]
0.4 -> [0.6, 0.4]
0.5 -> [0.5, 0.5]
0.8 -> [0.2, 0.8]
1.0 -> [0.0, 1.0]

The limit of this method, is you must pick a min and max.. you cannot represent numbers outside the range.


"""


def start_bin(input, lower=-20., upper=20., classes=256):
    k = (input - lower) / (upper - lower)
    k = k * (classes - 1)
    return k.floor().long()


def two_hot(input, lower=-20, upper=20, classes=256):
    """

    Encodes to two hot

    Numbers are represented on a line between min -- max
    The line is demarcated with C classes, and the two nearest
    classes to the number being represented hold values which sum to one
    the bucket closest to the number has the larger value

    ie: to represent the numbers between 0 .. 1 with 2 Classes...

    0.0 -> [1.0, 0.0]
    0.1 -> [0.9, 0.1]
    0.4 -> [0.6, 0.4]
    0.5 -> [0.5, 0.5]
    0.8 -> [0.2, 0.8]
    1.0 -> [0.0, 1.0]

    :param input: ( ...., 1 )
    :param lower: lower range of two hot
    :param upper: upper range of two hot
    :param classes: number of bins to use
    :return: (....., classes )
    """

    input = input.clamp(lower, upper)
    input_shape = input.shape
    input = input.flatten()
    b = torch.linspace(lower, upper, classes, device=input.device)
    b = torch.cat([b, torch.zeros(1, device=input.device)], dim=0)
    k = start_bin(input, lower, upper, classes)
    two_hot = torch.zeros(input.size(0), classes + 1, device=input.device)
    norm = (b[k + 1] - b[k]).abs()
    two_hot[torch.arange(input.size(0)), k] = (b[k + 1] - input).abs() / norm
    two_hot[torch.arange(input.size(0)), k + 1] = (b[k] - input).abs() / norm
    two_hot = two_hot[:, 0:-1]
    two_hot = two_hot.unflatten(0, input_shape[:-1])
    return two_hot


def decode_two_hot(twohot, lower=-20, upper=20, classes=256):
    """

    :param input: ( ...., classes )
    :param lower: lower range of two hot
    :param upper: upper range of two hot
    :param classes: number of bins to use
    :return: (....., 1 )
    """

    leading_dims = twohot.shape[0:-1]
    twohot = twohot.flatten(0, -2)
    values, indx = torch.topk(twohot, k=2, sorted=False)
    b = torch.linspace(lower, upper, classes, device=twohot.device)
    binsize = (upper - lower) / (classes - 1)
    b = torch.cat([b, torch.zeros(1, device=twohot.device)], dim=0)
    left = values[:, 0]
    base_value = b[indx[:, 0]]
    lower_residual = (1. - left) * binsize
    upper_residual = (left - 1.) * binsize
    is_lower = indx[:, 0] < indx[:, 1]
    decoded_two_hot = base_value + lower_residual * is_lower + upper_residual * ~is_lower
    return decoded_two_hot.unflatten(0, leading_dims).unsqueeze(-1)


def encode_range(input, upper, lower, classes):
    input = input.clamp(lower, upper)
    scale = (classes - 1) / (upper - lower)
    bias = lower
    return (input - bias) / scale


def decode_range(input, upper, lower, classes):
    scale = (classes - 1) / (upper - lower)
    bias = lower
    return scale * input + bias


def encode_two_hot_fast(input, classes):
    input_shape = input.shape
    input = input.flatten()
    two_hot = torch.zeros(input.size(0), classes + 1, device=input.device)
    k = torch.floor(input).long()
    remainder = torch.remainder(input, 1)
    two_hot[torch.arange(input.size(0)), k] = 1 - remainder
    two_hot[torch.arange(input.size(0)), k + 1] = remainder
    two_hot = two_hot[:, :-1]
    two_hot = two_hot.unflatten(0, input_shape[0:-1])
    return two_hot


def decode_two_hot_fast(input, classes):
    input = input.flatten(0, -2)
    k = input.argmax(-1)
    k_left = (k - 1)
    k_right = (k + 1).clamp(max=classes-1)
    padded_input = torch.cat((torch.zeros(input.size(0), 1), input), dim=1)
    value_k_left = padded_input[torch.arange(input.size(0)), k_left + 1]
    value_k_right = input[torch.arange(input.size(0)), k_right]
    left_greater = value_k_left.gt(value_k_right)
    right_greater = torch.logical_not(left_greater)
    major_weight = input[torch.arange(input.size(0)), k]
    minor_weight = torch.zeros(input.size(0))
    minor_weight[left_greater] = value_k_left[left_greater]
    minor_weight[right_greater] = value_k_right[right_greater]
    normalize_constant = major_weight + minor_weight
    major_weight = major_weight / normalize_constant
    minor_weight = minor_weight / normalize_constant
    major_value = k
    minor_value = torch.empty_like(k)
    minor_value[left_greater] = k_left[left_greater]
    minor_value[right_greater] = k_right[right_greater]
    return major_value * major_weight + minor_value * minor_weight


def encode_onehot(value, max_abs=8., bins=256):
    """
    param: value: (..., 1)
    param: max_abs -> the max and minimum value (symmetric)
    returns (..., bins )one_hot vectors rounded to the nearest bin
    """
    value = value.clamp(-max_abs, max_abs).squeeze(-1)
    value = value * ((bins - 1) // 2) / max_abs
    value = torch.round(value).long() + (bins - 1) // 2
    return one_hot(value, bins).float()


def decode_onehot(one_hot, max_abs=8., bins=256):
    """
    param: value: (..., bins)
    returns (..., 1)
    """
    value = one_hot.argmax(-1) - (bins - 1) // 2
    value = value * max_abs / ((bins - 1) // 2)
    return value.unsqueeze(-1)


def quantize(value, scale=8., bins=256):
    """
    param: value: (..., 1)
    param: max_abs -> the max and minimum value (symmetric)
    returns (..., bins )one_hot vectors rounded to the nearest bin
    """
    value = value.clamp(-scale, scale).squeeze(-1)
    value = value * ((bins - 1) // 2) / scale
    quantized_value = torch.round(value).long() + (bins - 1) // 2
    return quantized_value


def dequantize(quantized_value, scale=8., bins=256):
    """
    param: value: (..., bins)
    returns (..., 1)
    """
    value = quantized_value - (bins - 1) // 2
    value = value * scale / ((bins - 1) // 2)
    return value.unsqueeze(-1)


def encode_quantized_as_one_hot(quantized_value, bins=256):
    return one_hot(quantized_value, bins).float()


def decode_one_hot_as_quantized_value(onehot, bins=256):
    return onehot.argmax(-1)


def permute(index, permute_map):
    permute_map = permute_map.to(index.device)
    return permute_map[index]


def inv_permute(index, permute_map):
    inv_permute_map = torch.zeros_like(permute_map).to(index.device)
    inv_permute_map[permute_map] = torch.arange(permute_map.size(0)).to(index.device)
    return inv_permute_map[index]


class Codec:
    def __init__(self, encoder, decoder, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        self.params = kwargs

    def encode(self, data):
        return self.encoder(data, **self.params)

    def decode(self, data):
        return self.decoder(data, **self.params)


class CodecStack:
    def __init__(self, codecs):
        self.codecs = codecs

    def encode(self, data):
        for codec in self.codecs:
            data = codec.encode(data)
        return data

    def decode(self, data):
        for codec in reversed(self.codecs):
            data = codec.decode(data)
        return data


def make_codec(name):
    if name == 'onehot':
        return Codec(encoder=encode_onehot, decoder=decode_onehot, max_abs=1., bins=256)
    if name == 'symlog_onehot':
        permute_map_256 = torch.arange(256)
        permute_map_256[0] = 127
        permute_map_256[1:128] = torch.arange(127)
        return CodecStack([
            Codec(encoder=quantize, decoder=dequantize, scale=1., bins=256),
            Codec(encoder=permute, decoder=inv_permute, permute_map=permute_map_256),
            Codec(encoder=encode_quantized_as_one_hot, decoder=decode_one_hot_as_quantized_value, bins=256)
        ])
    if name == 'symlog':
        return Codec(encoder=symlog.symlog, decoder=symlog.symexp)


if __name__ == '__main__':
    values_in = torch.tensor([-8., -1, 0., 1., 8.]).unsqueeze(-1)
    values_in = torch.linspace(-8, 8, 1024).reshape(2, 512).unsqueeze(-1)
    one_hots = encode_onehot(values_in)
    values_out = decode_onehot(one_hots)
    assert torch.allclose(values_in, values_out, atol=0.05)

    values_in = torch.tensor([-1., -0.4, 0., 1., 1.]).unsqueeze(-1)
    values_in = torch.linspace(-1, 1, 1024).reshape(2, 512).unsqueeze(-1)
    one_hots = encode_onehot(values_in, max_abs=1.)
    values_out = decode_onehot(one_hots, max_abs=1.)
    assert torch.allclose(values_in, values_out, atol=0.05)

    one_hot_encoder = CodecStack([
        Codec(encoder=quantize, decoder=dequantize, scale=1., bins=256),
        Codec(encoder=encode_quantized_as_one_hot, decoder=decode_one_hot_as_quantized_value, bins=256)
    ])

    values_in = torch.linspace(-1, 1, 1024).reshape(2, 512).unsqueeze(-1)
    one_hots = one_hot_encoder.encode(values_in)
    values_out = one_hot_encoder.decode(one_hots)
    assert torch.allclose(values_in, values_out, atol=0.05)

    arng = torch.arange(10)
    permute_map = torch.tensor([5, 0, 1, 2, 3, 4, 6, 7, 8, 9])
    arng_permuted = permute(arng, permute_map)
    arng_unpermuted = inv_permute(arng_permuted, permute_map)
    assert torch.allclose(arng_unpermuted, arng)

    permute_map_256 = torch.arange(256)
    permute_map_256[0] = 127
    permute_map_256[1:128] = torch.arange(127)
    one_hot_encoder = CodecStack([
        Codec(encoder=quantize, decoder=dequantize, scale=1., bins=256),
        Codec(encoder=permute, decoder=inv_permute, permute_map=permute_map_256),
        Codec(encoder=encode_quantized_as_one_hot, decoder=decode_one_hot_as_quantized_value, bins=256)
    ])

    values_in = torch.linspace(-1, 1, 1024).reshape(2, 512).unsqueeze(-1)
    one_hots = one_hot_encoder.encode(values_in)
    values_out = one_hot_encoder.decode(one_hots)
    assert torch.allclose(values_in, values_out, atol=0.05)

    values_in = torch.tensor([-11, -5, 0., 5., 9])
    rescale = Codec(encoder=encode_range, decoder=decode_range, upper=9, lower=-11, classes=20)
    rescaled = rescale.encode(values_in)
    values_out = rescale.decode(rescaled)
    assert torch.allclose(values_in, values_out)

    values_in = torch.tensor([0., 0.5,  254.5, 255.]).unsqueeze(-1)
    values_in = torch.linspace(0., 255., 100).unsqueeze(-1)
    twohot = encode_two_hot_fast(values_in, classes=256)
    values_out = decode_two_hot_fast(twohot, classes=256)
    print(twohot)
    print(values_out)

    twohot_codec = CodecStack(codecs=[
        Codec(encoder=encode_range, decoder=decode_range, lower=-20, upper=20, classes=256),
        Codec(encoder=encode_two_hot_fast, decoder=decode_two_hot_fast, classes=256)
    ])
    values_in = torch.linspace(-20., 20, 100).unsqueeze(-1)
    twohot = twohot_codec.encode(values_in)
    values_out = twohot_codec.decode(twohot).unsqueeze(-1)
    for v_in, v_out in zip(values_in, values_out):
        print(v_in, v_out)
    assert(values_in.allclose(values_out))

    # lower = 0.
    # upper = 1.0
    # classes = 2
    #
    # N = classes * 3
    # input = torch.linspace(lower, upper, N).unsqueeze(-1)
    #
    # two_hot_vector = two_hot(input, lower, upper, classes)
    #
    # for inp, th in zip(input, two_hot_vector):
    #     print(inp.item(), th)
    #
    # decoded_two_hot = decode_two_hot(two_hot_vector, lower, upper, classes)
    # assert torch.allclose(input, decoded_two_hot)
    #
    # lower = -20.
    # upper = 20.
    # classes = 255
    #
    # N = classes * 3
    # input = torch.linspace(lower, upper, N).reshape(classes, 3).unsqueeze(-1)
    # two_hot_vector = two_hot(input, lower, upper, classes)
    # decoded_two_hot = decode_two_hot(two_hot_vector, lower, upper, classes)
    # print(input)
    # print(decoded_two_hot)
    # assert torch.allclose(input, decoded_two_hot)
    #
    #
    #
    # lower = 0.
    # upper = 9.
    # classes = 10
    #
    # N = classes * 3
    # input = torch.linspace(lower, upper, N).reshape(classes, 3).unsqueeze(-1)
    #
    # two_hot_vector = two_hot(input, lower, upper, classes)
    #
    # # for inp, th in zip(input.flatten(), two_hot_vector.flatten(0, 1)):
    # #     print(inp.item(), th)
