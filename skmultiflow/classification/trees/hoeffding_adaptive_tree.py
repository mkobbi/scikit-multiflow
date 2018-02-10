from abc import abstractmethod, ABCMeta
from random import random

import numpy as np
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.naive_bayes import NaiveBayes
from skmultiflow.classification.core.conditional_tests.instance_conditional_test import InstanceConditionalTest
from sklearn.preprocessing import normalize
import skmultiflow.classification.core.utils.utils
import skmultiflow.classification.core.utils.utils as utils

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
       # def calc_byte_size(self):
            #__sizeof__()
           # return super().calcByteSize() + int(SizeOf.sizeOf(self.children) + SizeOf.fullSizeOf(self.splitTest));

        def calc_byte_size_including_subtree(self):
            byte_size =   self.calc_byte_size_including_subtree()
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

        def ada_learning_node(self,initialClassObservations):
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

        def learn_from_instance(self,X,y,weight,ht : HoeffdingTree,parent,parentBranch):
            trueClass= y
            k= np.random.poisson(1.0)
            weightedInst = weight
            if k>0:
                weightedInst=weight * k;

            ClassPrediction = np.argmax(self.get_class_votes(X,y,weight, ht))
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

            self.learn_from_instance(X,trueClass,weightedInst,ht)

            weightSeen = self.get_weight_seen()

            if weightSeen - self._weight_seen_at_last_split_evaluation() >= ht.grace_period:
                ht._attempt_to_split(self, parent,parentBranch)
                self._weight_seen_at_last_split_evaluation(weightSeen)

        def normalize(self,doubles,sum):
            if sum== None:
                print("Can't normalize array. Sum is NaN ")
            elif sum==0:
                print("Can't normalize array. Sum is zero.")
            else:
                for i in range(doubles.length):
                    doubles[i] /= sum

        def get_class_votes(self ,X,ht):
            dist=0.0
            predictionOption= ht.leaf_prediction()
            if (predictionOption == 0):
                dist = self._observed_class_distribution
            elif(predictionOption == 1):
                dist=utils.do_naive_bayes_prediction(X,self.get_observed_class_distribution(),self._attribute_observers)
            else:
                if self._mc_correct_weight > self._nb_correct_weight:
                    dist=self._observed_class_distribution
                else:
                    dist=utils.do_naive_bayes_prediction(X,self.get_observed_class_distribution(),self._attribute_observers)
            distSum= np.sum(dist)
            if distSum * self.get_error_estimation() * self.get_error_estimation() > 0.0:
                normalize(dist, distSum * self.get_error_estimation() * self.get_error_estimation());

            return dist;



        def filter_instance_to_leaf(self,X,splitparent,parentBranch):
            self.




self.learn_from_instance(X, y, weight, ht)

            # Check for Split condition
            values, _= zip(*self.get_observed_class_distribution().items())
            weight_seen = self.get_weight_seen()
            if weight_seen - self.get_weight_seen_at_last_split_evaluation >= ht.grace_period:












