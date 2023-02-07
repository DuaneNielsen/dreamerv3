import torch

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


def two_hot(input, lower, upper, classes):
    """

    :param input: (N,
    :param lower:
    :param upper:
    :param classes:
    :return:
    """
    input = input.clamp(lower, upper)
    input_shape = input.shape
    input = input.flatten()
    b = torch.linspace(lower, upper, classes)
    b = torch.cat([b, torch.zeros(1)], dim=0)
    k = start_bin(input, lower, upper, classes)
    two_hot = torch.zeros(input.size(0), classes + 1)
    norm = (b[k + 1] - b[k]).abs()
    two_hot[torch.arange(N), k] = (b[k + 1] - input).abs() / norm
    two_hot[torch.arange(N), k + 1] = (b[k] - input).abs() / norm
    two_hot = two_hot[:, 0:-1]
    two_hot = two_hot.unflatten(0, input_shape)
    return two_hot


def decode_two_hot(twohot, lower, upper, classes):
    leading_dims = twohot.shape[0:-1]
    twohot = twohot.flatten(0, -2)
    values, indx = torch.topk(twohot, k=2, sorted=False)
    b = torch.linspace(lower, upper, classes)
    binsize = (upper - lower) / (classes - 1)
    b = torch.cat([b, torch.zeros(1)], dim=0)
    left = values[:, 0]
    base_value = b[indx[:, 0]]
    lower_residual = (1. - left) * binsize
    upper_residual = (left - 1.) * binsize
    is_lower = indx[:, 0] < indx[:, 1]
    decoded_two_hot = base_value + lower_residual * is_lower + upper_residual * ~is_lower
    return decoded_two_hot.unflatten(0, leading_dims)


if __name__ == '__main__':

    lower = 0.
    upper = 1.0
    classes = 2

    N = classes * 3
    input = torch.linspace(lower, upper, N)

    two_hot_vector = two_hot(input, lower, upper, classes)

    for inp, th in zip(input, two_hot_vector):
        print(inp.item(), th)

    decoded_two_hot = decode_two_hot(two_hot_vector, lower, upper, classes)
    assert torch.allclose(input, decoded_two_hot)

    lower = -20.
    upper = 20.
    classes = 255

    N = classes * 3
    input = torch.linspace(lower, upper, N).reshape(classes, 3)
    two_hot_vector = two_hot(input, lower, upper, classes)
    decoded_two_hot = decode_two_hot(two_hot_vector, lower, upper, classes)
    print(input)
    print(decoded_two_hot)
    assert torch.allclose(input, decoded_two_hot)



    lower = 0.
    upper = 9.
    classes = 10

    N = classes * 3
    input = torch.linspace(lower, upper, N).reshape(classes, 3)

    two_hot_vector = two_hot(input, lower, upper, classes)

    # for inp, th in zip(input.flatten(), two_hot_vector.flatten(0, 1)):
    #     print(inp.item(), th)