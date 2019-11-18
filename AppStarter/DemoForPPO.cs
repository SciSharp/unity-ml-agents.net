using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Unity3D.Trainers;
using static Tensorflow.Binding;
using static Tensorflow.Unity3D.Trainers.mock_brain;

namespace Tensorflow.Unity3D
{
    public class DemoForPPO
    {
        public bool Run()
        {
            tf.reset_default_graph();
            var graph = BuildGraph();

            return true;
        }

        public Graph BuildGraph()
        {
            var graph = tf.Graph().as_default();

            PPOModel model = null;

            tf_with(tf.variable_scope("FakeGraphScope"), delegate
            {
                model = new PPOModel(make_brain_parameters(discrete_action: true, visual_inputs: 0));

            });

            return graph;
        }
    }
}
