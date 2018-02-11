from abc import abstractmethod, ABCMeta
import random

import numpy as np
from skmultiflow.core.utils.utils import *
from abc import ABCMeta, abstractmethod
import math

import skmultiflow.classification.core.utils.utils as utils
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree

__author__ = 'Nehed ZOUAOUI, Mahmoud KOBBI, Jawher SOUDANI'
__version__ = '0.1'


class HoeffdingAdaptiveTree(HoeffdingTree):

    def get_purpose_string(self):
        return "Hoeffding Adaptive Tree for evolving data streams that uses ADWIN to replace branches for new ones."

    class NewNode(metaclass=ABCMeta):
        # Change for adwin
        # public boolean getErrorChange();
        @abstractmethod
        def number_leaves(self):
            raise NotImplementedError

        @abstractmethod
        def get_error_estimation(self):
            raise NotImplementedError

        @abstractmethod
        def get_error_width(self):
            raise NotImplementedError

        @abstractmethod
        def is_null_error(self):
            raise NotImplementedError

        @abstractmethod
        def kill_tree_childs(self, hat):
            raise NotImplementedError

        @abstractmethod
        def learn_from_instance(self, X, y, weight, ht, parent, parent_branch):
            raise NotImplementedError

        @abstractmethod
        def filter_instance_to_leaves(self, X, y, weight, myparent, parent_branch, found_nodes, update_splitter_counts):
            raise NotImplementedError

        def normalize(self, total, doubles):
            if total == np.NaN:
                print("Can't normalize array. Sum is NaN ")
            elif total == 0:
                print("Can't normalize array. Sum is zero.")
            else:
                for key, value in doubles.items():
                    value = value / total

        def max_index(self, doubles):
            maximum = 0
            argmax = 0
            for key, value in doubles.items():
                if value > maximum:
                    maximum = value
                    argmax = key
            return argmax

        def sum(self, doubles):
            sum = 0
            for i in range(doubles.length):
                sum += doubles[i]
            return sum

    class AdaSplitNode(HoeffdingTree.SplitNode, NewNode):

        def __init__(self, split_test, class_observations, size=-1):
            super().__init__(split_test, class_observations, size)
            self._alternate_tree = HoeffdingTree.Node()
            self._estimation_error_weight = ADWIN()
            self._error_change = False
            self._random_seed = 1
            self._classifier_random = random(self._random_seed)

        def calc_byte_size_including_subtree(self):
            byte_size = self.calc_byte_size_including_subtree()
            if self._alternate_tree is not None:
                byte_size += self._alternate_tree.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byte_size += self._estimation_error_weight.measure_byte_size()
            for child in self.children:
                if child is not None:
                    byte_size += child.calc_byte_size_including_subtree()
            return byte_size

        def number_leaves(self):
            num_leaves = 0
            for child in self.children:
                if child is not None:
                    num_leaves += child.number_leaves()
            return num_leaves

        def get_error_estimation(self):
            return self._estimation_error_weight._estimation

        def get_error_width(self):
            w = 0.0
            if not (self.is_null_error()):
                w = self._estimation_error_weight._width
            return w

        def is_null_error(self):
            return self._estimation_error_weight is None

        def kill_tree_childs(self, ht):
            for child in self._children:
                if child is not None:
                    if isinstance(child, HoeffdingAdaptiveTree.AdaSplitNode) and child._alterante_tree is not None:
                        child._alterante_tree.kill_tree_childs(ht)
                        ht._pruned_alterante_trees += 1

                    if isinstance(child, HoeffdingAdaptiveTree.AdaSplitNode):
                        child.kill_tree_childs(ht)

                    if isinstance(child, HoeffdingAdaptiveTree.ActiveLearningNode):
                        child = None
                        ht._active_leaf_node_cnt -= 1
                    elif isinstance(child, HoeffdingAdaptiveTree.InactiveLearningNode):
                        child = None
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
            ht: HoeffdingTree
                Hoeffding Tree to update.
            parent:
            parent_branch:

            """
            # true_class = instance.class_value()
            # New option wote
            # k = np.random.poisson(1)
            # if k>0:
            #    weight * k
            # Compute class_prediction using filter_instance_to_leaf
            class_prediction = 0
            if self.filter_instance_to_leaf(X, parent, parent_branch).node is not None:
                class_votes = self.filter_instance_to_leaf(X, parent, parent_branch).node.get_class_votes(X, ht)
                class_prediction = self.max_index(class_votes)
            bl_correct = (y == class_prediction)

            if self._estimation_error_weight is None:
                self._estimation_error_weight = ADWIN()

            old_error = self.get_error_estimation()
            self._estimation_error_weight.add_element(0.0 if bl_correct else 1.0)
            self._error_change = self._estimation_error_weight.detected_change()

            if self._error_change == True and old_error > self.get_error_estimation():
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
                    ht._switched_alterante_trees += 1
                elif bound < alt_error_rate - old_error_rate:
                    # Erase alternate tree
                    if isinstance(self._alternate_tree, HoeffdingTree.ActiveLearningNode):
                        self._alternate_tree = None
                    elif isinstance(self._alternate_tree, HoeffdingTree.InactiveLearningNode):
                        self._alternate_tree = None
                    else:
                        self._alternate_tree.kill_tree_childs(ht)
                    ht._pruned_alterate_trees += 1
            if self._alternate_tree is not None:
                self._alternate_tree.learn_from_instance(X, y, weight, hat, parent, parent_branch)
            child_branch = self.instance_child_index(X)
            child = self.get_child(child_branch)
            if child is not None:
                child.learn_from_instance(X, y, weight, hat, parent, parent_branch)

        def filter_instance_to_leaves(self, X, y, weight, myparent, parent_branch, update_splitter_counts,
                                      found_nodes=None):
            if update_splitter_counts:
                self._observed_class_distribution[y] += weight
            if found_nodes is None:
                found_nodes = []

            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    child.filter_instance_to_leaves(X, parent, parent_branch,
                                                    update_splitter_counts, found_nodes)
                else:
                    found_nodes.append(HoeffdingTree.FoundNode(None, self, child_index))
                if self._alternate_tree is not None:
                    self._alternate_tree.filter_instance_to_leaves(X, self, -999
                                                                   , update_splitter_counts, found_nodes)

    class AdaLearningNode(HoeffdingTree.LearningNodeNBAdaptive, NewNode):

        def __init__(self, initial_class_observations):

            HoeffdingTree.LearningNodeNBAdaptive.__init__(self, initial_class_observations)
            self._alternate_tree = HoeffdingTree.Node()
            self._estimation_error_weight = ADWIN()
            self._error_change = False
            self._random_seed = 1
            self.classifierRandom = random.seed(self._random_seed)

        def calc_byte_size_including_subtree(self):
            byte_size = self.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byte_size += self._estimation_error_weight.measure_byte_size()

            return byte_size

        def number_leaves(self):
            return 1

        def get_error_estimation(self):
            if self._estimation_error_weight is not None:
                return self._estimation_error_weight._estimation
            else:
                return 0

        def get_error_width(self):
            return self._estimation_error_weight._width

        def is_null_error(self):
            return self._estimation_error_weight is None

        def kill_tree_childs(self, ht):
            pass

        def learn_from_instance(self, X, y, weight, ht, parent=None, parentBranch=-1):
            true_class = y
            k = np.random.poisson(1.0, self.classifierRandom)
            weighted_inst = weight
            if k > 0:
                weighted_inst = weight * k
            # class_prediction = np.argmax(self.get_class_votes(X, y, weight, ht))
            # blCorrect = (trueClass == ClassPrediction)

            class_prediction = self.max_index(self.get_class_votes(X, ht))
            blCorrect = (y == class_prediction)
            if (self._estimation_error_weight is None):
                self._estimation_error_weight = ADWIN()
            # old_error = self.get_error_estimation()
            # blCorrect = (trueClass == ClassPrediction)

            old_error = self.get_error_estimation()
            # blCorrect = (trueClass == ClassPrediction)
            if (self._estimation_error_weight is None):
                self._estimation_error_weight = ADWIN()
            old_error = self.get_error_estimation()
            self._estimation_error_weight.add_element(0.0 if (blCorrect is True) else 1.0)

            self._error_change = self._estimation_error_weight.detected_change()
            if self._error_change == True and old_error > self.getErrorEstimation():
                self._error_change = False

            super().learn_from_instance(X, true_class, weighted_inst, ht)

            weight_seen = self.get_weight_seen()

            if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= ht.grace_period:
                ht._attempt_to_split(self, parent, parentBranch)
                self.set_weight_seen_at_last_split_evaluation(weight_seen)

        def get_class_votes(self, X, ht):
            dist = {}
            prediction_option = ht.leaf_prediction

            if prediction_option == 0:
                dist = self.get_observed_class_distribution()
            elif prediction_option == 1:
                dist = utils.do_naive_bayes_prediction(X, self.get_observed_class_distribution,
                                                       self._attribute_observers)
            elif self._mc_correct_weight > self._nb_correct_weight:
                dist = self.get_observed_class_distribution()
            else:
                dist = utils.do_naive_bayes_prediction(X, self._observed_class_distribution,
                                                       self._attribute_observers)
            dist_sum = sum(dist.values())

            if dist_sum * self.get_error_estimation() * self.get_error_estimation() > 0.0:
                self.normalize(dist_sum * self.get_error_estimation() * self.get_error_estimation(), dist)

            return dist

        def filter_instance_to_leaves(self, X, splitparent, parent_branch, update_splitter_counts, found_nodes=None):
            if found_nodes is None:
                found_nodes = []
            found_nodes.append(HoeffdingTree.FoundNode(self, splitparent, parent_branch))

    def __init__(self, *args, **kwargs):

        super(HoeffdingAdaptiveTree, self).__init__(*args, **kwargs)
        self._alternate_trees = 0
        self._pruned_alternate_trees = 0
        self._switch_alternate_trees = 0

    def _new_learning_node(self, initial_class_observations=None):
        return self.AdaLearningNode(initial_class_observations)

    # Override HoeffdingTree
    def new_split_node(self, split_test, class_observations, size):
        return self.AdaSplitNode(split_test, class_observations, size)

    def reset(self):
        self._alternate_trees = 0
        self._pruned_alternate_trees = 0
        self._switch_alternate_trees = 0

    # Override HoeffdingTree/BaseClassifier
    def partial_fit(self, X, y, classes=None, weight=None):
        super().partial_fit(X, y, classes=None, weight=None)

    # Override HoeffdingTree
    def _partial_fit(self, X, y, weight):

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        self._tree_root.learn_from_instance(X, y, weight, self, None, -1)

    def filter_instance_to_leaves(self, X, split_parent, parent_branch, update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, split_parent, parent_branch, update_splitter_counts, nodes)
        return np.asarray(nodes)

    # Override HoeffdingTree
    def get_votes_for_instance(self, X):
        if self._tree_root is not None:
            found_nodes = self.filter_instance_to_leaves(X=X, split_parent=None,
                                                         parent_branch=-1, update_splitter_counts=False)
            result = {}
            predict_path = 0
            for found_node in found_nodes:
                if found_node.parent_branch != -999:
                    leaf_node = found_node.node
                    if leaf_node is None:
                        leaf_node = found_node.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.update(dist)  # add elements to dictionary
            return result
        else:
            return {}

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
