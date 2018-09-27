import OpenEXR, Imath
import numpy as np


def load_exr(filename, dtype=np.float32):
    input_file = OpenEXR.InputFile(filename)
    pt = None
    if dtype == np.float32 :
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
    elif dtype == np.float16 :
        pt = Imath.PixelType(Imath.PixelType.HALF)
    elif dtype == np.uint :
        pt = Imath.PixelType(Imath.PixelType.UINT)
    else:
        print("Error - unknown dtype passed to load_exr")
        return None

    data_window = input_file.header()["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    r_bytes = input_file.channel('R', pt)
    g_bytes = input_file.channel('G', pt)
    b_bytes = input_file.channel('B', pt)
    #r_bytes, g_bytes, b_bytes = input_file.channels("RGB", pt)

    pixel_data = np.zeros((height, width, 3), dtype=dtype)
    pixel_data[:, :, 0] = np.fromstring(r_bytes, dtype=dtype).reshape(height, width)
    pixel_data[:, :, 1] = np.fromstring(g_bytes, dtype=dtype).reshape(height, width)
    pixel_data[:, :, 2] = np.fromstring(b_bytes, dtype=dtype).reshape(height, width)
    input_file.close()

    return pixel_data


def save_exr(filename, pixels, dtype=np.float32):
    pt = None
    if dtype == np.float32 :
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
    elif dtype == np.float16 :
        pt = Imath.PixelType(Imath.PixelType.HALF)
    elif dtype == np.uint :
        pt = Imath.PixelType(Imath.PixelType.UINT)
    else:
        print("Error - unknown dtype passed to load_exr")
        return None

    if pixels.dtype != dtype:
        pixels = pixels.astype(dtype)

    header = OpenEXR.Header(pixels.shape[1], pixels.shape[0])
    header['channels'] = {'R': Imath.Channel(pt),
                          'G': Imath.Channel(pt),
                          'B': Imath.Channel(pt)}
    outputFile = OpenEXR.OutputFile(filename, header)

    outputFile.writePixels(
        {'R': pixels[:, :, 0].tostring(), 'G': pixels[:, :, 1].tostring(), 'B': pixels[:, :, 2].tostring()})

    outputFile.close()
