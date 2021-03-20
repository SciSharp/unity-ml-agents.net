using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Unity3D.Trainers
{
    public class BrainParameters
    {
        public string brain_name;
        public int vector_observation_space_size;
        // public int num_stacked_vector_observations;
        public List<CameraResolution> camera_resolutions;
        public List<int> vector_action_space_size;
        public List<string> vector_action_descriptions;
        public string vector_action_space_type;
        public int number_visual_observations;

        public BrainParameters(string brain_name,
            int vector_observation_space_size,
            // int num_stacked_vector_observations,
            List<CameraResolution> camera_resolutions,
            List<int> vector_action_space_size,
            List<string> vector_action_descriptions,
            int vector_action_space_type_index)
        {
            this.brain_name = brain_name;
            this.vector_observation_space_size = vector_observation_space_size;
            // this.num_stacked_vector_observations = num_stacked_vector_observations;
            this.number_visual_observations = len(camera_resolutions);
            this.camera_resolutions = camera_resolutions;
            this.vector_action_space_size = vector_action_space_size;
            this.vector_action_descriptions = vector_action_descriptions;
            vector_action_space_type = (new string[] { "discrete", "continuous" })[vector_action_space_type_index];
        }
    }
}
