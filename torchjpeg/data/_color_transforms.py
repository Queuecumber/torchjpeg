class YChannel(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        if tensor.shape[0] == 3:
            tensor = tensor * 255
            tensor = 16. + tensor[0, :, :] * 65.481 / 255. + tensor[1, :, :] * 128.553 / 255. + tensor[2, :, :] * 24.966 / 255.
            tensor = tensor.unsqueeze(0).round() / 255.

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class YCbCr(object):
    def __init__(self):
        pass

    def __call__(self, pil_image):
        return pil_image.convert('YCbCr')

    def __repr__(self):
        return self.__class__.__name__ + '()'
