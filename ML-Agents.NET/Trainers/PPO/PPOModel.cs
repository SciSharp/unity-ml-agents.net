using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using static Tensorflow.Binding;
using NumSharp;

namespace Tensorflow.Unity3D.Trainers
{
    public class PPOModel : LearningModel
    {
        Tensor prev_action;
        Tensor all_log_probs;
        Tensor action_masks;
        Tensor output;
        Tensor normalized_logits;
        Tensor action_holder;
        Tensor action_oh;
        Tensor selected_actions;
        Tensor all_old_log_probs;
        Tensor old_normalized_logits;
        Tensor entropy;
        Tensor log_probs;
        Tensor old_log_probs;
        Tensor advantage;
        Tensor value_loss;
        Tensor policy_loss;

        /// <summary>
        /// Takes a Unity environment and model-specific hyper-parameters and returns the
        /// appropriate PPO agent model for the environment.
        /// </summary>
        /// <param name="brain">BrainInfo used to generate specific network graph.</param>
        /// <param name="lr">Learning rate.</param>
        /// <param name="lr_schedule">Learning rate decay schedule.</param>
        /// <param name="h_size">Size of hidden layers</param>
        /// <param name="epsilon">Value for policy-divergence threshold.</param>
        /// <param name="beta">Strength of entropy regularization.</param>
        /// <param name="max_step">Total number of training steps.</param>
        /// <param name="normalize">Whether to normalize vector observation input.</param>
        /// <param name="use_recurrent">Whether to use an LSTM layer in the network.</param>
        /// <param name="num_layers">Number of hidden layers between encoded input and policy & value layers</param>
        /// <param name="m_size">Size of brain memory.</param>
        /// <param name="seed">Seed to use for initialization of model.</param>
        /// <param name="stream_names">List of names of value streams. Usually, a list of the Reward Signals being used.</param>
        /// <param name="vis_encode_type"></param>
        public PPOModel(BrainParameters brain,
            float lr = 0.0001f,
            LearningRateSchedule lr_schedule = LearningRateSchedule.LINEAR,
            int h_size = 128,
            float epsilon = 0.2f,
            float beta = 0.001f,
            float max_step = 5e6f,
            bool normalize = false,
            bool use_recurrent = false,
            int num_layers = 2,
            int? m_size = null,
            int seed = 0,
            List<string> stream_names = null,
            EncoderType vis_encode_type = EncoderType.SIMPLE) : base(m_size: m_size,
                normalize: normalize,
                use_recurrent: use_recurrent,
                brain: brain,
                seed: seed,
                stream_names: stream_names)
        {
            if (num_layers < 1)
                num_layers = 1;
            if (brain.vector_action_space_type == "continuous")
            {
                throw new NotImplementedException("brain.vector_action_space_type");
            }
            else
            {
                create_dc_actor_critic(h_size, num_layers, vis_encode_type);
            }
            var learning_rate = create_learning_rate(lr_schedule, lr, global_step, max_step);
            create_losses(
                log_probs,
                old_log_probs,
                value_heads,
                entropy,
                beta,
                epsilon,
                lr,
                max_step);
        }

        /// <summary>
        /// Creates training-specific Tensorflow ops for PPO models.
        /// </summary>
        /// <param name="probs">Current policy probabilities</param>
        /// <param name="old_probs">Past policy probabilities</param>
        /// <param name="value_heads">Value estimate tensors from each value stream</param>
        /// <param name="entropy">Current policy entropy</param>
        /// <param name="beta">Entropy regularization strength</param>
        /// <param name="epsilon">Value for policy-divergence threshold</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="max_step">Total number of training steps.</param>
        private void create_losses(Tensor probs, Tensor old_probs, Dictionary<string, Tensor> value_heads, 
            Tensor entropy, float beta, float epsilon, 
            float lr, float max_step)
        {
            foreach(var name in value_heads.Keys)
            {
                throw new NotImplementedException("create_losses");
            }
            advantage = tf.placeholder(shape: -1, dtype: tf.float32, name: "advantages");
            advantage = tf.expand_dims(advantage, -1);
            var decay_epsilon = tf.train.polynomial_decay(
                epsilon, global_step, max_step, 0.1f, power: 1.0f);
            var decay_beta = tf.train.polynomial_decay(
                beta, global_step, max_step, 1e-5f, power: 1.0f);

            var value_losses = new List<Tensor>();
            foreach (var kp in value_heads)
            {
                var (name, head) = (kp.Key, kp.Value);
                throw new NotImplementedException("create_losses");
            }
            value_loss = tf.reduce_mean(value_losses.ToArray());
            var r_theta = tf.exp(probs - old_probs);
            var p_opt_a = r_theta * advantage;
            var clip = tf.clip_by_value(r_theta, 1.0f - decay_epsilon, 1.0f + decay_epsilon);
            var p_opt_b = clip * advantage;
            var min = tf.minimum(p_opt_a, p_opt_b);
            var partitions = tf.dynamic_partition(min, mask, 2);
            policy_loss = -tf.reduce_mean(partitions[1]);
        }

        /// <summary>
        /// Creates Discrete control actor-critic model.
        /// </summary>
        /// <param name="h_size">Size of hidden linear layers.</param>
        /// <param name="num_layers">Number of hidden linear layers.</param>
        /// <param name="vis_encode_type"></param>
        private void create_dc_actor_critic(int h_size,
            int num_layers,
            EncoderType vis_encode_type)
        {
            var hidden_streams = create_observation_streams(1, h_size, num_layers, vis_encode_type);
            var hidden = hidden_streams[0];

            if(use_recurrent)
            {
                prev_action = tf.placeholder(shape: (-1, len(act_size)), 
                    dtype: tf.int32, 
                    name: "prev_action");

                throw new NotImplementedException("create_dc_actor_critic use_recurrent");
            }

            var policy_branches = act_size.Select(size => tf.layers.dense(hidden,
                size,
                use_bias: false,
                kernel_initializer: scaled_init(0.01f))).ToArray();

            all_log_probs = tf.concat(policy_branches, axis: 1, name: "action_probs");
            action_masks = tf.placeholder(tf.float32, shape: (-1, sum(act_size)), name: "action_masks");
            var (output, _, normalized_logits) = create_discrete_action_masking_layer(all_log_probs, action_masks, act_size);
            output = tf.identity(output);
            normalized_logits = tf.identity(normalized_logits, name: "action");
            create_value_heads(stream_names, hidden);
            action_holder = tf.placeholder(shape: (-1, len(policy_branches)), 
                dtype: tf.int32, 
                name: "action_holder");
            var ah = action_holder[":", "0"];
            action_oh = tf.concat(range(len(act_size))
                .Select(i => tf.one_hot(ah, act_size[i]))
                .ToArray(), axis: 1);
            selected_actions = tf.stop_gradient(action_oh);
            all_old_log_probs = tf.placeholder(shape: (-1, sum(act_size)),
                dtype: tf.float32,
                name: "old_probabilities");

            var (_, _, old_normalized_logits) = create_discrete_action_masking_layer(all_old_log_probs, 
                action_masks, 
                act_size);

            var indice = np.cumsum(act_size.ToArray()).ToArray<int>().ToList();
            indice.Insert(0, 0);
            var action_idx = indice.ToArray();

            entropy = tf.reduce_sum(
                tf.stack
                (
                    range(len(act_size)).Select(i => tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels: tf.nn.softmax(all_log_probs[":", $"{action_idx[i]}:{action_idx[i + 1]}"]),
                        logits: all_log_probs[":", $"{action_idx[i]}:{action_idx[i + 1]}"])).ToArray(),
                    axis: 1
                ),
                axis: 1
            );

            log_probs = tf.reduce_sum(
                tf.stack
                (
                    range(len(act_size)).Select(i => -tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels: action_oh[":", $"{action_idx[i]}:{action_idx[i + 1]}"],
                        logits: normalized_logits[":", $"{action_idx[i]}:{action_idx[i + 1]}"])).ToArray(),
                    axis: 1
                ),
                axis: 1,
                keepdims: true
            );

            old_log_probs = tf.reduce_sum(
                tf.stack
                (
                    range(len(act_size)).Select(i => -tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels: action_oh[":", $"{action_idx[i]}:{action_idx[i + 1]}"],
                        logits: normalized_logits[":", $"{action_idx[i]}:{action_idx[i + 1]}"])).ToArray(),
                    axis: 1
                ),
                axis: 1,
                keepdims: true
            );
        }
    }
}
