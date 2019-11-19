using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Unity3D.Trainers;
using static Tensorflow.Binding;
using static UnitTests.mock_brain;

namespace UnitTests
{
    [TestClass]
    public class test_ppo
    {
        PPOModel model = null;
        Operation init = null;

        [TestMethod]
        public void test_ppo_model_dc_vector()
        {
            tf.reset_default_graph();

            tf_with(tf.variable_scope("FakeGraphScope"), delegate
            {
                model = new PPOModel(make_brain_parameters(discrete_action: true, visual_inputs: 0));
                init = tf.global_variables_initializer();
            });

            using (var sess = tf.Session())
            {
                sess.run(init);
                var results = sess.run
                    (
                        // run list
                        (model.output,
                        model.all_log_probs,
                        model.value,
                        model.entropy,
                        model.learning_rate),
                        // feed dict
                        (model.batch_size, 2),
                        (model.sequence_length, 1),
                        (model.vector_in, np.array(new int[,] { { 1, 2, 3, 1, 2, 3 }, { 3, 4, 5, 3, 4, 5 } })),
                        (model.action_masks, np.ones(new[] { 2, 2 }))
                    );

                print(results);
            }
        }
    }
}
