using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;
using System.Linq;
using Tensorflow.Operations.Activation;
using NumSharp;

namespace Tensorflow.Unity3D.Trainers
{
    public class LearningModel
    {
        protected int _version_number_ = 2;
        protected BrainParameters brain;

        protected RefVariable global_step;
        protected Tensor steps_to_increment;
        protected Tensor increment_step;
        public Tensor batch_size;
        public Tensor sequence_length;
        protected Tensor mask_input;
        protected Tensor mask;

        protected bool use_recurrent;
        protected int m_size;
        protected bool normalize;
        protected List<int> act_size;
        protected int vec_obs_size;
        protected int vis_obs_size;

        public Tensor vector_in;

        float EPSILON = 1e-7f;

        protected Dictionary<string, Tensor> value_heads = new Dictionary<string, Tensor>();
        public Tensor value;
        protected List<string> stream_names;

        public LearningModel(int? m_size = null,
            bool normalize = false,
            bool use_recurrent = false,
            BrainParameters brain = null,
            int seed = 0,
            List<string> stream_names = null)
        {
            tf.set_random_seed(seed);
            this.brain = brain;
            (global_step, increment_step, steps_to_increment) = create_global_steps();
            batch_size = tf.placeholder(shape: new int[0], dtype: tf.int32, name: "batch_size");
            sequence_length = tf.placeholder(shape: new int[0], dtype: tf.int32, name: "sequence_length");
            mask_input = tf.placeholder(shape: -1, dtype: tf.float32, name: "masks");
            mask = tf.cast(mask_input, tf.int32);
            this.stream_names = stream_names ?? new List<string>();
            this.use_recurrent = use_recurrent;
            m_size = use_recurrent ? m_size : 0;
            this.normalize = normalize;
            act_size = brain.vector_action_space_size;
            vec_obs_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations;
            vis_obs_size = brain.number_visual_observations;
            tf.Variable(
                Convert.ToInt32(brain.vector_action_space_type == "continuous"),
                name: "is_continuous_control",
                trainable: false,
                dtype: tf.int32
            );
            tf.Variable(
                _version_number_,
                name: "version_number",
                trainable: false,
                dtype: tf.int32
            );
            tf.Variable(m_size, name: "memory_size", trainable: false, dtype: tf.int32);
            if (brain.vector_action_space_type == "continuous")
                tf.Variable(
                    act_size[0],
                    name: "action_output_shape",
                    trainable: false,
                    dtype: tf.int32
                );
            else
                tf.Variable(
                    sum(act_size),
                    name: "action_output_shape",
                    trainable: false,
                    dtype: tf.int32
                );
        }

        private (RefVariable, Tensor, Tensor) create_global_steps()
        {
            var global_step = tf.Variable(0, name: "global_step", trainable: false, dtype: tf.int32);
            var steps_to_increment = tf.placeholder(shape:new int[0], dtype: tf.int32, name: "steps_to_increment");
            var increment_step = tf.assign(global_step, tf.add(global_step, steps_to_increment));
            return (global_step, increment_step, steps_to_increment);
        }

        /// <summary>
        /// Creates encoding stream for observations.
        /// </summary>
        /// <param name="num_streams">Number of streams to create.</param>
        /// <param name="h_size">Size of hidden linear layers in stream.</param>
        /// <param name="num_layers">Number of hidden linear layers in stream.</param>
        /// <param name="vis_encode_type"></param>
        /// <param name="stream_scopes"></param>
        /// <returns>List of encoded streams.</returns>
        public virtual List<Tensor> create_observation_streams(int num_streams,
            int h_size,
            int num_layers,
            EncoderType vis_encode_type = EncoderType.SIMPLE,
            List<string> stream_scopes = null)
        {
            IActivation activation_fn = tf.nn.swish();

            var visual_in = range(brain.number_visual_observations)
                .Select(i => create_visual_input(brain.camera_resolutions[i], name: $"visual_observation_{i}"))
                .ToArray();
            var vector_observation_input = create_vector_input();

            var final_hiddens = new List<Tensor>();
            Tensor hidden_state = null;
            Tensor final_hidden = null;
            Tensor hidden_visual = null;

            foreach (int i in range(num_streams))
            {
                var _scope_add = stream_scopes == null ? "" : stream_scopes[i];
                if(vis_obs_size > 0)
                {
                    switch (vis_encode_type)
                    {
                        case EncoderType.RESNET:
                            // create_resnet_visual_observation_encoder
                            break;
                        case EncoderType.NATURE_CNN:
                            // create_nature_cnn_visual_observation_encoder
                            break;
                        default:
                            // create_visual_observation_encoder
                            break;
                    }
                }
                if(brain.vector_observation_space_size > 0)
                {
                    hidden_state = create_vector_observation_encoder(
                        vector_observation_input,
                        h_size,
                        activation_fn,
                        num_layers,
                        scope: $"{_scope_add}main_graph_{i}",
                        reuse: false);
                }
                if (hidden_state != null && hidden_visual != null)
                    final_hidden = tf.concat(new[] { hidden_visual, hidden_state }, axis: 1);
                else if (hidden_state == null && hidden_visual != null)
                    final_hidden = hidden_visual;
                else if (hidden_state != null && hidden_visual == null)
                    final_hidden = hidden_state;
                else
                    throw new Exception("No valid network configuration possible. " +
                        "There are no states or observations in this brain");

                final_hiddens.append(final_hidden);
            }
            return final_hiddens;
        }

        public Tensor create_vector_observation_encoder(Tensor observation_input,
            int h_size,
            IActivation activation,
            int num_layers,
            string scope,
            bool reuse)
        {
            return tf_with(tf.variable_scope(scope), delegate
            {
                var hidden = observation_input;
                foreach(int i in range(num_layers))
                {
                    hidden = tf.layers.dense(
                        hidden,
                        h_size,
                        activation: activation,
                        reuse: reuse,
                        name: $"hidden_{i}",
                        kernel_initializer: tf.variance_scaling_initializer(1.0f)
                    );
                }

                return hidden;
            });
        }

        public Tensor create_visual_observation_encoder(Tensor image_input,
            int h_size
            )
        {
            throw new NotImplementedException("create_visual_observation_encoder");
        }

        public Tensor create_vector_input(string name = "vector_observation")
        {
            vector_in = tf.placeholder(shape: (-1, vec_obs_size),
                dtype: tf.float32,
                name: name);

            if (normalize)
            {
                throw new NotImplementedException("create_vector_input");
                //create_normalizer(vector_in);
                //return normalize_vector_obs(vector_in);
            }
            else
            {
                return vector_in;
            }
        }

        public Tensor swish(Tensor input_activation)
            => tf.multiply(input_activation, tf.sigmoid(input_activation));

        /// <summary>
        /// Creates image input op.
        /// </summary>
        /// <param name="camera_parameters">Parameters for visual observation from BrainInfo.</param>
        /// <param name="name">Desired name of input op.</param>
        /// <returns>input op.</returns>
        public Tensor create_visual_input(CameraResolution camera_parameters, string name)
        {
            var o_size_h = camera_parameters.height;
            var o_size_w = camera_parameters.width;
            var c_channels = camera_parameters.num_channels;

            var visual_in = tf.placeholder(shape: (-1, o_size_h, o_size_w, c_channels),
                dtype: tf.float32,
                name: name);
            return visual_in;
        }

        public IInitializer scaled_init(float scale)
            => tf.variance_scaling_initializer(factor: scale);

        public (Tensor, Tensor, Tensor) create_discrete_action_masking_layer(Tensor all_logits, 
            Tensor action_masks, List<int> action_size)
        {
            var actions = np.cumsum(action_size.ToArray())
                .ToArray<int>()
                .ToList();
            actions.Insert(0, 0);
            var action_idx = actions.ToArray();

            var branches_logits = range(len(action_size))
                .Select(i => all_logits[":", $"{action_idx[i]}:{action_idx[i + 1]}"])
                .ToArray();
            var branch_masks = range(len(action_size))
                .Select(i => action_masks[":", $"{action_idx[i]}:{action_idx[i + 1]}"])
                .ToArray();
            var raw_probs = range(len(action_size))
                .Select(k => tf.multiply(tf.nn.softmax(branches_logits[k]) + EPSILON, branch_masks[k]))
                .ToArray();
            var normalized_probs = range(len(action_size))
                .Select(k => tf.divide(raw_probs[k], tf.reduce_sum(raw_probs[k], axis: 1, keepdims: true)))
                .ToArray();
            var output = tf.concat(range(len(action_size))
                .Select(k => tf.multinomial(tf.log(normalized_probs[k] + EPSILON), 1))
                .ToArray(), axis: 1);

            return (
                output,
                tf.concat(range(len(action_size))
                        .Select(k => normalized_probs[k])
                        .ToArray(), axis: 1),
                tf.concat(range(len(action_size))
                        .Select(k => tf.log(normalized_probs[k] + EPSILON))
                        .ToArray(), axis: 1)
                );
        }

        public void create_value_heads(List<string> stream_names, Tensor hidden_input)
        {
            foreach(var name in stream_names)
            {
                var value = tf.layers.dense(hidden_input, 1, name: $"{name}_value");
                value_heads[name] = value;
            }
            value = tf.reduce_mean(list(value_heads.Values), 0);
        }

        public Tensor create_learning_rate(LearningRateSchedule lr_schedule, float lr, RefVariable global_step,
            float max_step)
        {
            /*if (lr_schedule == LearningRateSchedule.CONSTANT)
                learning_rate = tf.Variable(lr);
            else */if (lr_schedule == LearningRateSchedule.LINEAR)
                return tf.train.polynomial_decay(
                    lr, global_step, max_step, 1e-10f, power: 1.0f);
            throw new NotImplementedException("create_learning_rate");
        }
    }
}
