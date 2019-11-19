using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using System.Linq;
using Tensorflow.Unity3D.Trainers;

namespace UnitTests
{
    public static class mock_brain
    {
        public static BrainParameters make_brain_parameters(bool discrete_action = false,
            int visual_inputs = 0,
            bool stack = true,
            string brain_name = "RealFakeBrain",
            int vec_obs_size = 3)
        {
            var resolutions = range(visual_inputs)
                .Select(_ => new CameraResolution(width: 30, height: 40, num_channels: 3))
                .ToList();

            return new BrainParameters(brain_name: brain_name,
                vector_observation_space_size: vec_obs_size,
                num_stacked_vector_observations: stack ? 2 : 1,
                camera_resolutions: resolutions,
                vector_action_space_size: new List<int> { 2 },
                vector_action_descriptions: new List<string> { "", "" },
                vector_action_space_type_index: Convert.ToInt32(!discrete_action));
        }
    }
}
