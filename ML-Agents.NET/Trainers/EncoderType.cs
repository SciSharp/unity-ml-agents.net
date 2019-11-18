using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace Tensorflow.Unity3D.Trainers
{
    public enum EncoderType
    {
        [Description("simple")]
        SIMPLE = 1,

        [Description("nature_cnn")]
        NATURE_CNN = 2,

        [Description("resnet")]
        RESNET = 3
    }
}
