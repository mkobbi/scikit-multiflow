from abc import abstractmethod, ABCMeta
from random import random

import numpy as np

from skmultiflow.classification.core.driftdetection.adwin import ADWIN
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

    def getPurposeString(self):
        return "Hoeffding Adaptive Tree for evolving data streams that uses ADWIN to replace branches for new ones."

    class NewNode(metaclass=ABCMeta):
        # Change for adwin
        # public boolean getErrorChange();
        @abstractmethod
        def number_leaves(self): raise NotImplementedError

        @abstractmethod
        def get_error_estimation(self): raise NotImplementedError

        @abstractmethod
        def get_error_width(self): raise NotImplementedError

        @abstractmethod
        def is_null_error(self): raise NotImplementedError

        @abstractmethod
        def kill_tree_childs(self, hat): raise NotImplementedError

        @abstractmethod
        def learn_from_instance(self, X, y, weight, ht, parent, parent_branch): raise NotImplementedError

        @abstractmethod
        def filter_instance_to_leaves(self, X, y, weight, myparent, parent_branch, found_nodes, update_splitter_counts):
            raise NotImplementedError

    class AdaSplitNode(HoeffdingTree.SplitNode, NewNode):
        _alternate_tree = HoeffdingTree.Node()
        _estimation_error_weight = ADWIN()
        _error_change = False
        _random_seed = 1
        _classifier_random = random()

        # A revoir
       # def calc_byte_size(self):
            #__sizeof__()
           # return super().calcByteSize() + int(SizeOf.sizeOf(self.children) + SizeOf.fullSizeOf(self.splitTest));

        def __init__(self, split_test, class_observations, size=-1):
            super(self.AdaSplitNode, self).__init__(split_test, class_observations, size)
            self._classifier_random = random(seed=self._random_seed)

        def calc_byte_size_including_subtree(self):
            byte_size =   self.calc_byte_size_including_subtree()
            if self._alternate_tree is not None:
                byte_size += self._alternate_tree.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byte_size += self._estimation_error_weight.()
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
                    if isinstance(child, self.AdaSplitNode) and child._alterante_tree is not None:
                        child._alterante_tree.kill_tree_childs(ht)
                        ht._pruned_alterante_trees += 1
                    # Recursive delete of SplitNode
                    if isinstance(child, self.AdaSplitNode):
                        child.kill_tree_childs(ht)
                    if isinstance(child, super.ActiveLearningNode):
                        # child = None <- this is unreachable
                        ht._active_leaf_node_cnt -= 1
                    elif isinstance(child, super.InactiveLearningNode):
                        # child = None <- this is unreachable
                        ht._inactive_leaf_node_cnt -= 1

        def learn_from_instance(self, X, y, weight, ht, parent, parent_branch):
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

            """
            # true_class = instance.class_value()
            # New option wote
            k = np.random.poisson(1)
            # if k>0:
            #    weight * k
            # Compute class_prediction using filter_instance_to_leaf
            class_prediction = 0
            if self.filter_instance_to_leaf(X, parent, parent_branch).node is not None:
                class_votes = self.filter_instance_to_leaves(X, parent, parent_branch).node.get_class_votes(X, ht)
                class_prediction = max(class_votes, key=class_votes.get)
            bl_correct = y == class_prediction

            if self._estimation_error_weight is None:
                self._estimation_error_weight = ADWIN()

            old_error = self.get_error_estimation()
            self._error_change = self._estimation_error_weight.add_element(0.0 if bl_correct else 1.0)

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
                    if isinstance(self._alternate_tree, super.ActiveLearningNode):
                        self._alternate_tree = None
                    elif isinstance(self._alternate_tree, super.InactiveLearningNode):
                        self._alternate_tree = None
                    else:
                        self._alternate_tree.kill_tree_childs(ht)
                    ht._pruned_alterate_trees += 1

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
                    found_nodes.append(super.FoundNode(None, self, child_index))
                if self._alternate_tree is not None:
                    self._alternate_tree.filter_instance_to_leaves(X, y, weight, self, -999, found_nodes
                                                                   , update_splitter_counts)


    class AdaLearningNode(HoeffdingTree.LearningNodeNBAdaptive, NewNode):
        _alternate_tree = HoeffdingTree.Node()
        _estimation_error_weight = ADWIN()
        _error_change = False
        _random_seed = 1
        _classifier_random = random()

        def calc_byte_size_including_subtree(self):
            byteSize = HoeffdingTree.LearningNodeNBAdaptive.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byteSize += self._estimation_error_weight.measureByteSize()

            return byteSize

        def ada_learning_node(self, initialClassObservations):
            self.initialClassObservations
            self._classifier_random = random(self._random_seed)

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
            self._estimation_error_weight == None

        def kill_tree_childs(self, ht):
            pass

        def learn_from_instance(self, X, y, weight, ht: HoeffdingTree, parent, parentBranch):
            trueClass = y
            k = np.random.poisson(1.0)
            weightedInst = weight
            if k > 0:
                weightedInst = weight * k;

            ClassPrediction = np.argmax(self.get_class_votes(X, y, weight, ht))
            blCorrect = (trueClass == ClassPrediction)
            if (self._estimation_error_weight is not None):
                self._estimation_error_weight = ADWIN()

            oldError = self.getErrorEstimation()
            blCorrect = (trueClass == ClassPrediction)
            if (self._estimation_error_weight == None):
                self._estimation_error_weight = ADWIN()
            oldError = self.getErrorEstimation()
            self._error_change = self._estimation_error_weight.detected_change()
            if (self._error_change == True and oldError > self.getErrorEstimation()):
                self._error_change = False

            self.learn_from_instance(X, trueClass, weightedInst, ht)

            weightSeen = self.get_weight_seen()

            if weightSeen - self._weight_seen_at_last_split_evaluation() >= ht.grace_period:
                ht._attempt_to_split(self, parent, parentBranch)
                self._weight_seen_at_last_split_evaluation(weightSeen)

        def normalize(self, doubles, sum):
            if sum == None:
                print("Can't normalize array. Sum is NaN ")
            elif sum == 0:
                print("Can't normalize array. Sum is zero.")
            else:
                for i in range(doubles.length):
                    doubles[i] /= sum

        def get_class_votes(self, X, ht):
            dist = 0.0
            predictionOption = ht.leaf_prediction()
            if (predictionOption == 0):
                dist = self._observed_class_distribution
            elif (predictionOption == 1):
                dist = utils.do_naive_bayes_prediction(X, self.get_observed_class_distribution(),
                                                       self._attribute_observers)
            else:
                if self._mc_correct_weight > self._nb_correct_weight:
                    dist = self._observed_class_distribution
                else:
                    dist = utils.do_naive_bayes_prediction(X, self.get_observed_class_distribution(),
                                                           self._attribute_observers)
            distSum = np.sum(dist)
            if distSum * self.get_error_estimation() * self.get_error_estimation() > 0.0:
                self.normalize(dist, distSum * self.get_error_estimation() * self.get_error_estimation());

            return dist;

        def filter_instance_to_leaf(self, X, splitparent, parentBranch):
            foundNodes = []
            foundNodes.append(self.FoundNode(X, splitparent, parentBranch))
