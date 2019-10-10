
# Usage:
# Parse the YUV and convert it into RGB
# size = (height, width) # height and then width
# videoObj = VideoCaptureYUV(image, size, isGrayScale=is_gray_str.__contains__('Y'))
# ret, yuv, rgb = videoObj.getYUVAndRGB()


import numpy as np

class VideoCaptureYUV:
    def __init__(self, filename, size, isGrayScale = False):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.stream = open(filename, 'rb')
        self.yuv = None
        self.rgb = None
        # Calculate the actual image size in the stream (accounting for rounding
        # of the resolution)
        # self.fwidth = (self.width + 31) // 32 * 32
        # self.fheight = (self.height + 15) // 16 * 16

        self.fwidth = self.width 
        self.fheight = self.height
        self.isGrayScale = isGrayScale

        self.Y = None
        self.U = None
        self.V = None

    def seek_n_frames(n):
        stream.seek(int(n * self.width * self.height * 1.5))

    # close stream?

    def read_yuv(self):
        try:
            # Load the Y (luminance) data from the stream
            Y = np.fromfile(self.stream, dtype=np.uint8, count=self.fwidth*self.fheight).\
            reshape((self.fheight, self.fwidth))


            # Load the UV (chrominance) data from the stream, and double its size            
            if self.isGrayScale:
                U = np.ones([self.fheight//2, self.fwidth//2], dtype=np.uint8).\
                reshape((self.fheight//2, self.fwidth//2)).\
                repeat(2, axis=0).repeat(2, axis=1)*128

                V = np.ones([self.fheight//2, self.fwidth//2], dtype=np.uint8).\
                reshape((self.fheight//2, self.fwidth//2)).\
                repeat(2, axis=0).repeat(2, axis=1)*128
            else:
                U = np.fromfile(self.stream, dtype=np.uint8, count=(self.fwidth//2)*(self.fheight//2)).\
                reshape((self.fheight//2, self.fwidth//2)).\
                repeat(2, axis=0).repeat(2, axis=1)

                V = np.fromfile(self.stream, dtype=np.uint8, count=(self.fwidth//2)*(self.fheight//2)).\
                reshape((self.fheight//2, self.fwidth//2)).\
                repeat(2, axis=0).repeat(2, axis=1)
            
             # Stack the YUV channels together, crop the actual resolution, convert to
            # floating point for later calculations, and apply the standard biases
            YUV = np.dstack((Y, U, V))[:self.height, :self.width, :].astype(np.float)
            YUV[:, :, 0]  = YUV[:, :, 0]  - 16   # Offset Y by 16
            YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
            self.yuv = YUV
            self.Y = Y
            self.U = U
            self.V = V

        except Exception as e:
            print ('Reading Error')
            return False
        return True

    def convertYUVtoRGB(self):
        # YUV conversion matrix from ITU-R BT.601 version (SDTV)
        # Note the swapped R and B planes!
        #              Y       U       V
        M = np.array([[1.164,  2.017,  0.000],    # B
                  [1.164, -0.392, -0.813],    # G
                  [1.164,  0.000,  1.596]])   # R
        # Take the dot product with the matrix to produce BGR output, clamp the
        # results to byte range and convert to bytes
        self.rgb = self.yuv.dot(M.T).clip(0, 255).astype(np.uint8)



    def setGray(self):
        if self.yuv is None:
            print('Function Error')
            return

        U = np.ones([self.fheight//2, self.fwidth//2], dtype=np.uint8).\
                reshape((self.fheight//2, self.fwidth//2)).\
                repeat(2, axis=0).repeat(2, axis=1)*128
        V = np.ones([self.fheight//2, self.fwidth//2], dtype=np.uint8).\
                reshape((self.fheight//2, self.fwidth//2)).\
                repeat(2, axis=0).repeat(2, axis=1)*128

        YUV = np.dstack((self.Y, U, V))[:self.height, :self.width, :].astype(np.float)
        YUV[:, :, 0]  = YUV[:, :, 0]  - 16   # Offset Y by 16
        YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
        
        M = np.array([[1.164,  2.017,  0.000],    # B
                  [1.164, -0.392, -0.813],    # G
                  [1.164,  0.000,  1.596]])   # R
        # Take the dot product with the matrix to produce BGR output, clamp the
        # results to byte range and convert to bytes
        rgb = YUV.dot(M.T).clip(0, 255).astype(np.uint8)

        return YUV, rgb

    def getYUV(self):
        return self.yuv

    def getRGB(self):
        return self.rgb


    def getYUVAndRGB(self):
        
        ret = self.read_yuv()
        if not ret:
            return ret, None, None
        self.convertYUVtoRGB()

        return ret, self.yuv, self.rgb


