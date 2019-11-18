using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Unity3D.Trainers
{
    public class CameraResolution
    {
        public int height;
        public int width;
        public int num_channels;

        public CameraResolution(int height, 
            int width, 
            int num_channels)
        {
            this.height = height;
            this.width = width;
            this.num_channels = num_channels;
        }
    }
}
