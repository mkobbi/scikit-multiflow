from abc import abstractmethod, ABCMeta
from random import random

import numpy as np

import skmultiflow.classification.core.utils.utils as utils
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
# from skmultiflow.classification.trees.hoeffding_adaptive_tree import HoeffdingAdaptiveTree
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree

__author__ = 'Nehed ZOUAOUI, Mahmoud KOBBI, Jawher SOUDANI'
__version__ = '0.1'


class HoeffdingAdaptiveTree(HoeffdingTree):
    """
    Hoeffding Adaptive Tree for evolving data streams.

    Notes
    -----
    This adaptive Hoeffding Tree uses ADWIN to monitor performance of
    branches on the tree and to replace them with new branches when their
    accuracy decreases if the new branches are more accurate.
    See details in:
    <p>Adaptive Learning from Evolving Data Streams. Albert Bifet, Ricard Gavaldà.
    IDA 2009


    Same parameters as HoeffdingTreeNBAdaptive
    -l : Leaf prediction to use: MajorityClass (MC), Naive Bayes (NB) or NaiveBayes
    adaptive (NBAdaptive).


    """

    def __init__(self, max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                 split_criterion='info_gain', split_confidence=0.0000001, tie_threshold=0.05, binary_split=False,
                 stop_mem_management=False, remove_poor_atts=False, no_preprune=False, leaf_prediction='mc',
                 nb_threshold=0, nominal_attributes=None):
        super().__init__(max_byte_size, memory_estimate_period, grace_period, split_criterion, split_confidence,
                         tie_threshold, binary_split, stop_mem_management, remove_poor_atts, no_preprune,
                         leaf_prediction, nb_threshold, nominal_attributes)
        self._alternate_trees = 0
        self._pruned_alternate_trees = 0
        self._switched_alternate_trees = 0

    @staticmethod
    def get_purpose_string():
        return "Hoeffding Adaptive Tree for evolving data streams that uses ADWIN to replace branches for new ones."

    class NewNode(metaclass=ABCMeta):
        # Change for adwin
        # public boolean getErrorChange();
        @abstractmethod
        @property
        def number_leaves(self): raise NotImplementedError

        @abstractmethod
        @property
        def get_error_estimation(self): raise NotImplementedError

        @abstractmethod
        @property
        def get_error_width(self): raise NotImplementedError

        @abstractmethod
        def is_null_error(self): raise NotImplementedError

        @abstractmethod
        def kill_tree_childs(self, hat): raise NotImplementedError

        @abstractmethod
        def learn_from_instance(self, X, y, weight, ht, parent=None, parent_branch=-1): raise NotImplementedError

        @abstractmethod
        def filter_instance_to_leaves(self, X, y, weight, myparent, parent_branch, found_nodes, update_splitter_counts):
            raise NotImplementedError

    class AdaSplitNode(HoeffdingTree.SplitNode, NewNode):

        # A revoir
        # def calc_byte_size(self):
        # __sizeof__()
        # return super().calcByteSize() + int(SizeOf.sizeOf(self.children) + SizeOf.fullSizeOf(self.splitTest));

        def __init__(self, split_test, class_observations, size=-1):
            super(HoeffdingAdaptiveTree.AdaSplitNode, self).__init__(split_test, class_observations, size)
            self._alternate_tree = HoeffdingTree.Node()
            self._estimation_error_weight = ADWIN()
            self._error_change = False
            self._random_seed = 1
            self._classifier_random = random.Random()
            self._classifier_random.seed(seed=self._random_seed)

        def calc_byte_size_including_subtree(self):
            byte_size = self.calc_byte_size_including_subtree()
            if self._alternate_tree is not None:
                byte_size += self._alternate_tree.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byte_size += self._estimation_error_weight.measure_byte_size()
            for child in self.children:
                if child is not None:
                    byte_size += child.calcByteSizeIncludingSubtree()
            return byte_size

        def number_leaves(self):
            num_leaves = 0
            for child in self.children:
                if child is not None:
                    num_leaves += child.number_leaves()
            return num_leaves

        def get_error_estimation(self):
            return self._estimation_error_weight._estimation()

        def get_error_width(self):
            w = 0.0
            if not (self.is_null_error()):
                w = self._estimation_error_weight._width()
            return w

        def is_null_error(self):
            return self._estimation_error_weight is None

        def kill_tree_childs(self, ht):
            for child in self._children:
                if child is not None:
                    # Delete alternate tree if it exists
                    if isinstance(child, HoeffdingAdaptiveTree.AdaSplitNode) and child._alterante_tree is not None:
                        child._alterante_tree.kill_tree_childs(ht)
                        ht._pruned_alterante_trees += 1
                    # Recursive delete of SplitNode
                    if isinstance(child, HoeffdingAdaptiveTree.AdaSplitNode):
                        child.kill_tree_childs(ht)
                    if isinstance(child, HoeffdingAdaptiveTree.ActiveLearningNode):
                        # child = None <- this is unreachable
                        ht._active_leaf_node_cnt -= 1
                    elif isinstance(child, HoeffdingTree.InactiveLearningNode):
                        # child = None <- this is unreachable
                        ht._inactive_leaf_node_cnt -= 1

        def learn_from_instance(self, X, y, weight, ht, parent=None, parent_branch=-1):
            """Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: HoeffdingAdaptiveTree
                Hoeffding Adaptive Tree to update.
            parent:
            parent_branch:

            """
            # true_class = instance.class_value()
            # New option wote
            # k = np.random.poisson(1)
            # if k>0:
            #    weight * k
            # Compute class_prediction using filter_instance_to_leaf
            # class_prediction = 0
            if self.filter_instance_to_leaf(X, parent, parent_branch).node is not None:
                class_votes = self.filter_instance_to_leaf(X, parent, parent_branch).node.get_class_votes(X, ht)
                class_prediction = max(class_votes, key=class_votes.get)
                bl_correct = y == class_prediction

                if self._estimation_error_weight is None:
                    self._estimation_error_weight = ADWIN()

                old_error = self.get_error_estimation()
                self._error_change = self._estimation_error_weight.add_element(0.0 if bl_correct else 1.0)

                if self._error_change and old_error > self.get_error_estimation():
                    # if error is decreasing, don't do anything
                    self._error_change = False

                # Check condition to build a new alternate tree
                if self._error_change is True:
                    # Start a new alternative tree: learning node
                    self._alternate_tree = ht._new_learning_node()
                    ht._alternate_trees += 1
                elif self._alternate_tree is not None and self._alternate_tree.is_null_error() is False:
                    old_error_rate = self.get_error_estimation()
                    alt_error_rate = self._alternate_tree.get_error_estimation()
                    f_delta = 0.05
                    f_n = 1.0 / self._alternate_tree.get_error_width() + 1.0 / self.get_error_width()
                    bound = np.sqrt(2.0 * old_error_rate + (1.0 - old_error_rate) * np.log(2.0 / f_delta) * f_n)
                    if bound < old_error_rate - alt_error_rate:
                        # Switch alternate tree
                        ht._active_leaf_node_cnt -= self.number_leaves()
                        ht._active_leaf_node_cnt += self._alternate_tree.number_leaves()
                        self.kill_tree_childs(ht)
                        if parent is not None:
                            parent.set_child(parent_branch, self._alternate_tree)
                        else:
                            # Switch root tree
                            ht._tree_root = ht._tree_root._alternate_tree
                        ht._switched_alternate_trees += 1
                    elif bound < alt_error_rate - old_error_rate:
                        # Erase alternate tree
                        if isinstance(self._alternate_tree, HoeffdingTree.ActiveLearningNode):
                            self._alternate_tree = None
                        elif isinstance(self._alternate_tree, HoeffdingTree.InactiveLearningNode):
                            self._alternate_tree = None
                        else:
                            self._alternate_tree.kill_tree_childs(ht)
                        ht._pruned_alternate_trees += 1

        def filter_instance_to_leaves(self, X, y, weight, myparent, parent_branch, found_nodes, update_splitter_counts):
            if update_splitter_counts:
                self._observed_class_distribution[y] += weight
            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    child.filter_instance_to_leaves(X, y, weight, self, child_index, found_nodes,
                                                    update_splitter_counts)
                else:
                    found_nodes.append(HoeffdingTree.FoundNode(None, self, child_index))
                if self._alternate_tree is not None:
                    # self._alternate_tree.__class__ = HoeffdingAdaptiveTree
                    self._alternate_tree = HoeffdingAdaptiveTree.AdaSplitNode(
                        self._alternate_tree._split_test,
                        self._alternate_tree._observed_class_distribution)
                    self._alternate_tree.filter_instance_to_leaves(X, y, weight, self, -999, found_nodes
                                                                   , update_splitter_counts)

    class AdaLearningNode(HoeffdingTree.LearningNodeNBAdaptive, NewNode):

        def calc_byte_size_including_subtree(self):
            byte_size = HoeffdingTree.LearningNodeNBAdaptive.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byte_size += self._estimation_error_weight.measure_byte_size()

            return byte_size

        def __init__(self, initial_class_observations):
            super(HoeffdingAdaptiveTree.AdaLearningNode, self).__init__(initial_class_observations)
            self._random_seed = 1
            self._classifier_random = random.Random()
            self._classifier_random.seed(self._random_seed)
            self._alternate_tree = HoeffdingTree.Node()
            self._estimation_error_weight = ADWIN()
            self._error_change = False

        def number_leaves(self):
            return 1

        def get_error_estimation(self):
            if self._estimation_error_weight is not None:
                return self._estimation_error_weight._estimation()
            else:
                return 0

        def get_error_width(self):
            return self._estimation_error_weight._width()

        def is_null_error(self):
            return self._estimation_error_weight is None

        def kill_tree_childs(self, ht):
            pass

        def learn_from_instance(self, X, y, weight, ht, parent=None, parent_branch=-1):
            """Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: HoeffdingTree
                Hoeffding Tree to update.
            parent: SplitNode
                Always equal to None. Needed for compatibility issues
            parent_branch: int
                Always equal to -1. Needed for compatibility issues


            """
            true_class = y
            k = np.random.poisson(1.0)
            weighted_inst = weight
            if k > 0:
                weighted_inst = weight * k

            # class_prediction = np.argmax(self.get_class_votes(X, y, weight, ht))
            # blCorrect = (trueClass == ClassPrediction)
            if self._estimation_error_weight is not None:
                self._estimation_error_weight = ADWIN()

            # old_error = self.get_error_estimation()
            # blCorrect = (trueClass == ClassPrediction)
            if self._estimation_error_weight is None:
                self._estimation_error_weight = ADWIN()
            old_error = self.get_error_estimation()
            self._error_change = self._estimation_error_weight.detected_change()
            if self._error_change is True and old_error > self.getErrorEstimation():
                self._error_change = False

            self.learn_from_instance(X, true_class, weighted_inst, ht)

            weight_seen = self.get_weight_seen()

            if weight_seen - self._weight_seen_at_last_split_evaluation() >= ht.grace_period:
                ht._attempt_to_split(self, parent, parent_branch)
                self._weight_seen_at_last_split_evaluation(weight_seen)

        @staticmethod
        def normalize(doubles, total):
            if total is None:
                print("Can't normalize array. Sum is NaN ")
            elif total == 0:
                print("Can't normalize array. Sum is zero.")
            else:
                for i in range(doubles.length):
                    doubles[i] /= total

        def get_class_votes(self, X, ht):
            # dist = 0.0
            prediction_option = ht.leaf_prediction()
            if prediction_option == 0:
                dist = self._observed_class_distribution
            elif prediction_option == 1:
                dist = utils.do_naive_bayes_prediction(X, self.get_observed_class_distribution(),
                                                       self._attribute_observers)
            else:
                if self._mc_correct_weight > self._nb_correct_weight:
                    dist = self._observed_class_distribution
                else:
                    dist = utils.do_naive_bayes_prediction(X, self.get_observed_class_distribution(),
                                                           self._attribute_observers)
            dist_sum = np.sum(dist)
            total = dist_sum * self.get_error_estimation() * self.get_error_estimation()
            if total > 0.0:
                HoeffdingAdaptiveTree.AdaLearningNode.normalize(dist, total)

            return dist

        def filter_instance_to_leaves(self, X, y, weight, split_parent, parent_branch, found_nodes,
                                      update_splitter_counts):
            found_nodes.append(self.FoundNode(X, split_parent, parent_branch))

    def new_learning_node(self, initial_class_observations=None):
        return self._new_learning_node(initial_class_observations)

    def _new_learning_node(self, initial_class_observations=None):
        return self.AdaLearningNode(initial_class_observations)

    def new_split_node(self, split_test, class_observations, size=-1):
        return self.AdaSplitNode(split_test, class_observations, size)

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        """
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: Not used.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.
        """
        self._partial_fit(X, y, weight)

    def _partial_fit(self, X, y, weight):
        """Incrementally trains the model. Train samples (instances) are compossed of X attributes and their
        corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. iI not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for the instance and update the leaf node
          statistics.
        * If growth is allowed and the number of instances that the leaf has observed between split attempts
          exceed the grace period then attempt to split.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        if self._tree_root is None:
            self._tree_root = self.new_learning_node()
            self._active_leaf_node_count = 1
        self._tree_root.learn_from_instance(X, y, weight, self, None, -1)

    def predict(self, X):
        return super().predict(X)

    def get_votes_for_instance(self, X):
        if self._tree_root is not None:
            found_nodes = self.filter_instance_to_leaves(X=X, y=None, weight=None, parent=None,
                                                         parent_branch=-1, update_splitter_counts=False)
            result = []
            # prediction_paths = 0
            for found_node in found_nodes:
                if found_node.parent_branch != -999:
                    leaf_node = found_node.node
                    if leaf_node is None:
                        leaf_node = found_node.parent
                    dist = leaf_node.get_class_votes(X)
                    # Albert: changed for weights
                    # distSum = np.sum(dist);
                    # if (distSum > 0.0):
                    # np.normalize(dist,distSum)
                    result.append(dist)
                    # prediction_paths + +
                    # if (prediction_paths > self.max_prediction_paths):
                    # self.max_prediction_paths++;
            return result
        return []

    def predict_proba(self, X):
        return self.get_votes_for_instance(X)

    def score(self, X, y):
        raise NotImplementedError

    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch, update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, y, weight, parent, parent_branch,
                                                  nodes, update_splitter_counts)
        return np.asarray(nodes)
