using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace Tensorflow.Unity3D.Trainers
{
    public enum LearningRateSchedule
    {
        [Description("constant")]
        CONSTANT = 0,

        [Description("linear")]
        LINEAR = 1
    }
}
