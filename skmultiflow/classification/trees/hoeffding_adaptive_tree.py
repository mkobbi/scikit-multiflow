from abc import abstractmethod, ABCMeta
from random import random

from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree

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


     */
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
        def learn_from_instance(self, instance, hat, parent, parent_branch): raise NotImplementedError

        @abstractmethod
        def filter_instance_to_leaves(self, instance, myparent, parent_branch, found_nodes, update_splitter_counts):
            raise NotImplementedError

    class AdaSplitNode(HoeffdingTree.SplitNode, NewNode):
        _alternate_tree = HoeffdingTree.Node()
        _estimation_error_weight = ADWIN()
        _error_change = False
        _randomSeed = 1
        _classifier_random = random()

        # A revoir
        def calc_byte_size(self):
            __sizeof__()
            return super().calcByteSize() + int(SizeOf.sizeOf(self.children) + SizeOf.fullSizeOf(self.splitTest));

        def calc_byte_size_including_subtree(self):
            byte_size = calc_byte_size()
            if self._alternate_tree is not None:
                byte_size += self._alternate_tree.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byte_size += self._estimation_error_weight.measureByteSize()
            for child in self.children:
                if child is not None:
                    byte_size += child.calcByteSizeIncludingSubtree()
            return byte_size

        def number_leaves(self):
            pass

        def get_error_estimation(self):
            pass

        def get_error_width(self):
            pass

        def is_null_error(self):
            pass

        def kill_tree_childs(self, hat):
            pass

        def learn_from_instance(self, instance, hat, parent, parent_branch):
            pass

        def filter_instance_to_leaves(self, instance, myparent, parent_branch, found_nodes, update_splitter_counts):
            pass
