from skmultiflow.classification.trees.hoeffding_adaptive_tree import HoeffdingAdaptiveTree
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption

dataset = "covtype"

# 1. Create a stream

opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/" + dataset + ".csv", "CSV", False)
stream = FileStream(opt, -1, 1)
# 2. Prepare for use
stream.prepare_for_use()
# 2. Instantiate the HoeffdingTree classifier
h = HoeffdingAdaptiveTree()
# 3. Setup the evaluator
eval = EvaluatePrequential(pretrain_size=1000, output_file='result_' + dataset + '.csv', max_instances=10000,
                           batch_size=1, n_wait=500, max_time=1000000000, task_type='classification', show_plot=True,
                           plot_options=['performance'])
# 4. Run
eval.eval(stream=stream, classifier=h)
