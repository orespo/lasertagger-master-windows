import pandas

import spacy

from edit_distance import SequenceMatcher
from tqdm import tqdm


import heapq

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np

from itertools import chain, combinations, permutations
from multiprocessing import Pool
from progressbar import ProgressBar, SimpleProgress
from tqdm import tqdm

import sys
import threading


try:
    import thread
except ImportError:
    import _thread as thread


# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SARI score for evaluating paraphrasing and other text generation models.
The score is introduced in the following paper:
   Optimizing Statistical Machine Translation for Text Simplification
   Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen and Chris Callison-Burch
   In Transactions of the Association for Computational Linguistics (TACL) 2015
   http://cs.jhu.edu/~napoles/res/tacl2016-optimizing.pdf
This implementation has two differences with the GitHub [1] implementation:
  (1) Define 0/0=1 instead of 0 to give higher scores for predictions that match
      a target exactly.
  (2) Fix an alleged bug [2] in the deletion score computation.
[1] https://github.com/cocoxu/simplification/blob/master/SARI.py
    (commit 0210f15)
[2] https://github.com/cocoxu/simplification/issues/6
"""

import collections

import numpy as np


import matplotlib.pyplot as plt
import networkx as nx
import re


class Decomposition(object):
    def __init__(self, decomposition_list):
        self.decomposition_list = [str(step) for step in decomposition_list]

    def _get_graph_edges(self):
        edges = []
        for i, step in enumerate(self.decomposition_list):
            references = [int(x) for x in re.findall(r"@@(\d)@@", step)]
            step_edges = [(i+1, ref) for ref in references]
            edges.extend(step_edges)

        return edges

    def to_string(self):
        return " @@SEP@@ ".join([x.replace("  ", " ").strip() for x in self.decomposition_list])

    def to_graph(self, nodes_only=False):
        # initiate a directed graph
        graph = nx.DiGraph()

        # add edges
        if nodes_only:
            edges = []
        else:
            edges = self._get_graph_edges()
        graph.add_edges_from(edges)

        # add nodes
        nodes = self.decomposition_list
        for i in range(len(nodes)):
            graph.add_node(i+1, label=nodes[i])

        # handle edge cases where artificial nodes need to be added
        for node in graph.nodes:
            if 'label' not in graph.nodes[node]:
                graph.add_node(node, label='')

        return graph

    def draw_decomposition(self):
        graph = self.to_graph(False)
        draw_decomposition_graph(graph)


def draw_decomposition_graph(graph, title=None):
    options = {
        'node_color': 'lightblue',
        'node_size': 400,
        'width': 1,
        'arrowstyle': '-|>',
        'arrowsize': 14,
    }

    pos = nx.spring_layout(graph, k=0.5)
    nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, **options)
    for node in graph.nodes:
        plt.text(pos[node][0], pos[node][1]-0.1,
                 s=graph.nodes[node]['label'],
                 bbox=dict(facecolor='red', alpha=0.5),
                 horizontalalignment='center',
                 wrap=True)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.show()
    plt.clf()


def get_decomposition_from_string(decomposition_str):
    decomposition = [d.strip() for d in decomposition_str.split("@@SEP@@")]
    return Decomposition(decomposition)


def get_decomposition_from_tokens(decomposition_tokens):
    decomposition_str = ' '.join(decomposition_tokens)
    return get_decomposition_from_string(decomposition_str)

# import tensorflow as tf

# The paper that intoduces the SARI score uses only the precision of the deleted
# tokens (i.e. beta=0). To give more emphasis on recall, you may set, e.g.,
# beta=1.
BETA_FOR_SARI_DELETION_F_MEASURE = 0


def _get_ngram_counter(ids, n):
  """Get a Counter with the ngrams of the given ID list.
  Args:
    ids: np.array or a list corresponding to a single sentence
    n: n-gram size
  Returns:
    collections.Counter with ID tuples as keys and 1s as values.
  """
  # Remove zero IDs used to pad the sequence.
  ids = [token_id for token_id in ids if token_id != 0]
  ngram_list = [tuple(ids[i:i + n]) for i in range(len(ids) + 1 - n)]
  ngrams = set(ngram_list)
  counts = collections.Counter()
  for ngram in ngrams:
    counts[ngram] = 1
  return counts


def _get_fbeta_score(true_positives, selected, relevant, beta=1):
  """Compute Fbeta score.
  Args:
    true_positives: Number of true positive ngrams.
    selected: Number of selected ngrams.
    relevant: Number of relevant ngrams.
    beta: 0 gives precision only, 1 gives F1 score, and Inf gives recall only.
  Returns:
    Fbeta score.
  """
  precision = 1
  if selected > 0:
    precision = true_positives / selected
  if beta == 0:
    return precision
  recall = 1
  if relevant > 0:
    recall = true_positives / relevant
  if precision > 0 and recall > 0:
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)
  else:
    return 0


def get_addition_score(source_counts, prediction_counts, target_counts):
  """Compute the addition score (Equation 4 in the paper)."""
  added_to_prediction_counts = prediction_counts - source_counts
  true_positives = sum((added_to_prediction_counts & target_counts).values())
  selected = sum(added_to_prediction_counts.values())
  # Note that in the paper the summation is done over all the ngrams in the
  # output rather than the ngrams in the following set difference. Since the
  # former does not make as much sense we compute the latter, which is also done
  # in the GitHub implementation.
  relevant = sum((target_counts - source_counts).values())
  return _get_fbeta_score(true_positives, selected, relevant)


def get_keep_score(source_counts, prediction_counts, target_counts):
  """Compute the keep score (Equation 5 in the paper)."""
  source_and_prediction_counts = source_counts & prediction_counts
  source_and_target_counts = source_counts & target_counts
  true_positives = sum((source_and_prediction_counts &
                        source_and_target_counts).values())
  selected = sum(source_and_prediction_counts.values())
  relevant = sum(source_and_target_counts.values())
  return _get_fbeta_score(true_positives, selected, relevant)


def get_deletion_score(source_counts, prediction_counts, target_counts, beta=0):
  """Compute the deletion score (Equation 6 in the paper)."""
  source_not_prediction_counts = source_counts - prediction_counts
  source_not_target_counts = source_counts - target_counts
  true_positives = sum((source_not_prediction_counts &
                        source_not_target_counts).values())
  selected = sum(source_not_prediction_counts.values())
  relevant = sum(source_not_target_counts.values())
  return _get_fbeta_score(true_positives, selected, relevant, beta=beta)


def get_sari_score(source_ids, prediction_ids, list_of_targets,
                   max_gram_size=4, beta_for_deletion=0):
  """Compute the SARI score for a single prediction and one or more targets.
  Args:
    source_ids: a list / np.array of SentencePiece IDs
    prediction_ids: a list / np.array of SentencePiece IDs
    list_of_targets: a list of target ID lists / np.arrays
    max_gram_size: int. largest n-gram size we care about (e.g. 3 for unigrams,
        bigrams, and trigrams)
    beta_for_deletion: beta for deletion F score.
  Returns:
    the SARI score and its three components: add, keep, and deletion scores
  """
  addition_scores = []
  keep_scores = []
  deletion_scores = []
  for n in range(1, max_gram_size + 1):
    source_counts = _get_ngram_counter(source_ids, n)
    prediction_counts = _get_ngram_counter(prediction_ids, n)
    # All ngrams in the targets with count 1.
    target_counts = collections.Counter()
    # All ngrams in the targets with count r/num_targets, where r is the number
    # of targets where the ngram occurs.
    weighted_target_counts = collections.Counter()
    num_nonempty_targets = 0
    for target_ids_i in list_of_targets:
      target_counts_i = _get_ngram_counter(target_ids_i, n)
      if target_counts_i:
        weighted_target_counts += target_counts_i
        num_nonempty_targets += 1
    for gram in weighted_target_counts.keys():
      weighted_target_counts[gram] /= num_nonempty_targets
      target_counts[gram] = 1
    keep_scores.append(get_keep_score(source_counts, prediction_counts,
                                      weighted_target_counts))
    deletion_scores.append(get_deletion_score(source_counts, prediction_counts,
                                              weighted_target_counts,
                                              beta_for_deletion))
    addition_scores.append(get_addition_score(source_counts, prediction_counts,
                                              target_counts))

  avg_keep_score = sum(keep_scores) / max_gram_size
  avg_addition_score = sum(addition_scores) / max_gram_size
  avg_deletion_score = sum(deletion_scores) / max_gram_size
  sari = (avg_keep_score + avg_addition_score + avg_deletion_score) / 3.0
  return sari, avg_keep_score, avg_addition_score, avg_deletion_score


def get_sari(source_ids, prediction_ids, target_ids, max_gram_size=4):
  """Computes the SARI scores from the given source, prediction and targets.
  Args:
    source_ids: A 2D tf.Tensor of size (batch_size , sequence_length)
    prediction_ids: A 2D tf.Tensor of size (batch_size, sequence_length)
    target_ids: A 3D tf.Tensor of size (batch_size, number_of_targets,
        sequence_length)
    max_gram_size: int. largest n-gram size we care about (e.g. 3 for unigrams,
        bigrams, and trigrams)
  Returns:
    A 4-tuple of 1D float Tensors of size (batch_size) for the SARI score and
        the keep, addition and deletion scores.
  """

  # def get_sari_numpy(source_ids, prediction_ids, target_ids):
  """Iterate over elements in the batch and call the SARI function."""
  sari_scores = []
  keep_scores = []
  add_scores = []
  deletion_scores = []
  # Iterate over elements in the batch.
  for source_ids_i, prediction_ids_i, target_ids_i in zip(
      source_ids, prediction_ids, target_ids):
    sari, keep, add, deletion = get_sari_score(
        source_ids_i, prediction_ids_i, target_ids_i, max_gram_size,
        BETA_FOR_SARI_DELETION_F_MEASURE)
    sari_scores.append(sari)
    keep_scores.append(keep)
    add_scores.append(add)
    deletion_scores.append(deletion)
  return (np.asarray(sari_scores), np.asarray(keep_scores),
          np.asarray(add_scores), np.asarray(deletion_scores))

  # sari, keep, add, deletion = tf.py_func(
  #     get_sari_numpy,
  #     [source_ids, prediction_ids, target_ids],
  #     [tf.float64, tf.float64, tf.float64, tf.float64])
  # return sari, keep, add, deletion


def quit_function(fn_name):
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()
    # raises KeyboardInterrupt
    thread.interrupt_main()


def exit_after(s):
    """
    use as decorator to exit process if
    function takes longer than s seconds
    """
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

class GraphMatchScorer(object):
    def __init__(self):
        self.sequence_matcher = SequenceMatchScorer(remove_stop_words=False)

    def node_subst_cost_lexical(self, node1, node2):
        return 1 - self.sequence_matcher.get_match_score(node1['label'], node2['label'])

    @exit_after(600)
    def normalized_graph_edit_distance(self, graph1, graph2, structure_only):
        """Returns graph edit distance normalized between [0,1].
        Parameters
        ----------
        graph1 : graph
        graph2 : graph
        structure_only : whether to use node substitution cost 0 (e.g. all nodes are identical).
        Returns
        -------
        float
            The normalized graph edit distance of G1,G2.
            Node substitution cost is normalized string edit distance of labels.
            Insertions cost 1, deletion costs 1.
        """
        if structure_only:
            node_subst_cost = lambda x, y: 0
        else:
            node_subst_cost = self.node_subst_cost_lexical
        approximated_distances = nx.optimize_graph_edit_distance(graph1, graph2,
                                                                 node_subst_cost=node_subst_cost)
        total_cost_graph1 = len(graph1.nodes) + len(graph1.edges)
        total_cost_graph2 = len(graph2.nodes) + len(graph2.edges)
        normalization_factor = max(total_cost_graph1, total_cost_graph2)

        dist = None
        for v in approximated_distances:
            dist = v

        return float(dist)/normalization_factor

    def get_edit_distance_match_scores(self, predictions, targets, structure_only=False):
        distances = []
        num_examples = len(predictions)
        for i in tqdm(range(num_examples)):
            try:
                dist = self.normalized_graph_edit_distance(predictions[i], targets[i],
                                                           structure_only)
            except KeyboardInterrupt:
                print(f"skipping example: {i}")
                dist = None

            distances.append(dist)

        return distances


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)

    def count(self):
        return len(self.elements)


class AStarSearcher(object):
    def __init__(self):
        self.sequence_matcher = SequenceMatchScorer(remove_stop_words=False)
        self.graph1 = None
        self.graph2 = None

    def _get_label(self, graph_number, node):
        assert graph_number in [1, 2]
        if graph_number == 1:
            return self.graph1.nodes[node]['label']
        else:
            return self.graph2.nodes[node]['label']

    def _get_edit_ops(self, edit_path):
        edit_ops = set()
        graph1_matched_nodes, graph2_matched_nodes = self._get_matched_nodes(edit_path)
        graph1_unmatched_nodes = [node for node in self.graph1.nodes if node not in graph1_matched_nodes]
        graph2_unmatched_nodes = [node for node in self.graph2.nodes if node not in graph2_matched_nodes]

        if not edit_path:
            i = 1
        else:
            i = min(graph1_unmatched_nodes)

        subsets1 = self._get_all_subsets(graph1_unmatched_nodes)
        subsets2 = self._get_all_subsets(graph2_unmatched_nodes)

        # for i in graph1_unmatched_nodes:
        # add {v_i -> u_j}, {v_i -> u_j+u_j+1}, ...
        for subset in subsets2:
            edit_ops.add(((i,), subset))

        # add {v_i -> del}
        edit_ops.add(((i,), (-1,)))

        # add {v_i+v_i+1 -> u_j}, ...
        for subset in subsets1:
            if i in subset:
                for j in graph2_unmatched_nodes:
                    edit_ops.add((subset, (j,)))

        return list(edit_ops)

    @staticmethod
    def _get_edge_crossing_cost(edit_path):
        segments = []
        for edit_op in edit_path:
            for source in edit_op[0]:
                for target in edit_op[1]:
                    segments.append((source, target))

        cost = 0.0
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                # all segments have the same orientation (going down from source to target).
                if (segments[i][0] < segments[j][0] and segments[i][1] > segments[j][1]) or \
                        (segments[i][0] > segments[j][0] and segments[i][1] < segments[j][1]):
                    cost += 1.0

        return cost

    def _get_merge_op_cost(self, merge_nodes, target):
        graph2_label = self._get_label(2, target)
        min_merge_cost = 1
        for permutation in permutations(merge_nodes):
            graph1_labels = [self._get_label(1, node) for node in permutation]
            graph1_label = ' '.join(graph1_labels)
            permutation_merge_cost = 1 - self.sequence_matcher.get_match_score(graph1_label, graph2_label)

            if permutation_merge_cost < min_merge_cost:
                min_merge_cost = permutation_merge_cost

        return min_merge_cost

    def _get_split_op_cost(self, source, split_nodes):
        graph1_label = self._get_label(1, source)
        min_split_cost = 1
        for permutation in permutations(split_nodes):
            graph2_labels = [self._get_label(2, node) for node in permutation]
            graph2_label = ' '.join(graph2_labels)
            permutation_split_cost = 1 - self.sequence_matcher.get_match_score(graph1_label, graph2_label)

            if permutation_split_cost < min_split_cost:
                min_split_cost = permutation_split_cost

        return min_split_cost

    def _get_edit_path_cost(self, edit_path):
        cost = 0

        for edit_op in edit_path:
            # node insertion
            if edit_op[0] == (-1,):
                cost += 1

            # node deletion
            elif edit_op[1] == (-1,):
                cost += 1

            # node substitution
            elif len(edit_op[0]) == len(edit_op[1]):
                graph1_label = self._get_label(1, edit_op[0][0])
                graph2_label = self._get_label(2, edit_op[1][0])
                substitution_cost = 1 - self.sequence_matcher.get_match_score(graph1_label, graph2_label)
                cost += substitution_cost

            # node merging
            elif len(edit_op[0]) > 1:
                min_merge_cost = self._get_merge_op_cost(edit_op[0], edit_op[1][0])
                cost += min_merge_cost * len(edit_op[0])

            # node splitting
            elif len(edit_op[1]) > 1:
                min_split_cost = self._get_split_op_cost(edit_op[0][0], edit_op[1])
                cost += min_split_cost * len(edit_op[1])

            else:
                raise RuntimeError(
                    "get_edit_op_cost: edit op does not match any edit type: {}".format(edit_op)
                )

        edge_crossing_cost = self._get_edge_crossing_cost(edit_path)

        return cost + edge_crossing_cost

    def _get_heuristic_cost(self, edit_path):
        # graph1_curr_nodes, graph2_curr_nodes = self._get_matched_nodes(edit_path)
        # heuristic_cost = num_graph1_unmatched_nodes + num_graph2_unmatched_nodes
        # num_graph1_unmatched_nodes = graph1.number_of_nodes() - len(graph1_curr_nodes)
        # num_graph2_unmatched_nodes = graph2.number_of_nodes() - len(graph2_curr_nodes)

        return 0

    def _get_curr_edit_path_string(self, edit_path):
        result = []

        for edit_op in edit_path:
            source = ','.join([self._get_label(1, node) if node != -1 else '-' for node in edit_op[0]])
            target = ','.join([self._get_label(2, node) if node != -1 else '-' for node in edit_op[1]])
            result.append('[{}]->[{}]'.format(source, target))

        return ', '.join(result)

    def _is_isomorphic_graphs(self):
        nm = iso.categorical_node_match('label', '')
        em = iso.numerical_edge_match('weight', 1)
        return nx.is_isomorphic(self.graph1, self.graph2,
                                node_match=nm, edge_match=em)

    @staticmethod
    def get_edit_op_count(edit_path):
        edit_op_counts = {
            "insertion": 0,
            "deletion": 0,
            "substitution": 0,
            "merging": 0,
            "splitting": 0
        }

        for edit_op in edit_path:
            if edit_op[0] == (-1,):
                edit_op_counts["insertion"] += 1

            elif edit_op[1] == (-1,):
                edit_op_counts["deletion"] += 1

            elif len(edit_op[1]) > 1:
                edit_op_counts["splitting"] += 1

            elif len(edit_op[0]) == len(edit_op[1]) == 1:
                edit_op_counts["substitution"] += 1

            elif len(edit_op[0]) > 1:
                edit_op_counts["merging"] += 1

            else:
                raise RuntimeError("_get_edit_op_type: edit op type was not identified: {}".format(edit_op))

        return edit_op_counts

    @staticmethod
    def _get_all_subsets(ss):
        subsets = chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))
        subsets = [subset for subset in subsets if len(subset) > 0]

        return subsets

    @staticmethod
    def _get_matched_nodes(edit_path):
        graph1_matched_nodes, graph2_matched_nodes = [], []
        for (graph1_nodes, graph2_nodes) in edit_path:
            graph1_matched_nodes.extend([node for node in graph1_nodes if node != -1])
            graph2_matched_nodes.extend([node for node in graph2_nodes if node != -1])

        return graph1_matched_nodes, graph2_matched_nodes

    def set_graphs(self, graph1, graph2):
        self.graph1 = graph1
        self.graph2 = graph2

    def a_star_search(self, debug=False):
        assert self.graph1 and self.graph2

        if self._is_isomorphic_graphs():
            self.graph1, self.graph2 = None, None
            return [], "", 0, 0, 0, 0

        found_best_path = False
        queue = PriorityQueue()
        edit_ops = self._get_edit_ops([])
        for edit_op in edit_ops:
            queue.put([edit_op],
                      self._get_edit_path_cost([edit_op]) +
                      self._get_heuristic_cost([edit_op]))
        num_ops = len(edit_ops)

        while True:
            if queue.empty():
                raise RuntimeError("a_star_search: could not find a complete edit path.")

            curr_cost, curr_edit_path = queue.get()
            graph1_curr_nodes, graph2_curr_nodes = self._get_matched_nodes(curr_edit_path)

            if len(graph1_curr_nodes) < self.graph1.number_of_nodes():
                edit_ops = self._get_edit_ops(curr_edit_path)
                for edit_op in edit_ops:
                    curr_edit_path_extended = curr_edit_path + [edit_op]
                    queue.put(curr_edit_path_extended,
                              self._get_edit_path_cost(curr_edit_path_extended) +
                              self._get_heuristic_cost(curr_edit_path_extended))
                num_ops += len(edit_ops)

            elif len(graph2_curr_nodes) < self.graph2.number_of_nodes():
                edit_ops = [((-1,), (node,)) for node in self.graph2.nodes
                            if node not in graph2_curr_nodes]
                curr_edit_path_extended = curr_edit_path + edit_ops
                queue.put(curr_edit_path_extended,
                          self._get_edit_path_cost(curr_edit_path_extended) +
                          self._get_heuristic_cost(curr_edit_path_extended))
                num_ops += len(edit_ops)

            elif debug:
                if not found_best_path:
                    found_best_path = True
                    best_edit_path, best_cost = curr_edit_path, curr_cost
                    num_paths_true = queue.count() + 1
                    num_ops_true = num_ops
                    num_ops = 0
                elif not queue.empty():
                    continue
                else:
                    best_edit_path_string = self._get_curr_edit_path_string(best_edit_path)
                    explored_ops_ratio = (num_ops_true * 100.0) / (num_ops_true + num_ops)
                    # print("explored {:.2f}% of the ops.".format(explored_ops_ratio))
                    self.graph1, self.graph2 = None, None
                    return best_edit_path, best_edit_path_string, best_cost, num_paths_true, num_ops_true,\
                           explored_ops_ratio

            else:
                curr_edit_path_string = self._get_curr_edit_path_string(curr_edit_path)
                num_paths = queue.count() + 1   # +1 for the current path we just popped

                self.graph1, self.graph2 = None, None
                return curr_edit_path, curr_edit_path_string, curr_cost, num_paths, num_ops, None


def get_ged_plus_score(idx, graph1, graph2, exclude_thr, debug):
    if exclude_thr and \
            (graph1.number_of_nodes() > exclude_thr or
             graph2.number_of_nodes() > exclude_thr):
        print(f"skipping example: {idx}")
        return idx, None, None, None, None, None, None

    a_start_searcher = AStarSearcher()
    a_start_searcher.set_graphs(graph1, graph2)
    curr_edit_path, _, curr_cost, curr_num_paths, curr_num_ops, curr_ops_ratio = \
        a_start_searcher.a_star_search(debug=debug)
    curr_edit_op_counts = a_start_searcher.get_edit_op_count(curr_edit_path)

    return idx, curr_edit_path, curr_cost, curr_num_paths, curr_num_ops,\
           curr_ops_ratio, curr_edit_op_counts


def get_ged_plus_scores(decomposition_graphs, gold_graphs,
                        exclude_thr=None, debug=False, num_processes=5):
    samples = list(zip(decomposition_graphs, gold_graphs))
    pool = Pool(num_processes)
    pbar = ProgressBar(widgets=[SimpleProgress()], maxval=len(samples)).start()

    results = []
    _ = [pool.apply_async(get_ged_plus_score,
                          args=(i, samples[i][0], samples[i][1], exclude_thr, debug),
                          callback=results.append)
         for i in range(len(samples))]
    while len(results) < len(samples):
        pbar.update(len(results))
    pbar.finish()
    pool.close()
    pool.join()

    edit_op_counts = {
        "insertion": 0,
        "deletion": 0,
        "substitution": 0,
        "merging": 0,
        "splitting": 0
    }
    idxs, scores_tmp, num_paths, num_ops, ops_ratio = [], [], [], [], []
    for result in results:
        idx, curr_edit_path, curr_cost, curr_num_paths, curr_num_ops, curr_ops_ratio, curr_edit_op_counts = result
        idxs.append(idx)
        scores_tmp.append(curr_cost)
        if not curr_cost:
            continue

        num_paths.append(float(curr_num_paths))
        num_ops.append(float(curr_num_ops))
        if debug:
            ops_ratio.append(curr_ops_ratio)

        for op in curr_edit_op_counts:
            edit_op_counts[op] += curr_edit_op_counts[op]

    scores = [score for (idx, score) in sorted(zip(idxs, scores_tmp))]

    print("edit op statistics:", edit_op_counts)
    print("number of explored paths: mean {:.2}, min {:.2}, max {:.2}".format(
        np.mean(num_paths), np.min(num_paths), np.max(num_paths)))
    print("number of edit ops: mean {:.2}, min {:.2}, max {:.2}".format(
        np.mean(num_ops), np.min(num_ops), np.max(num_ops)))
    if debug:
        print("explored ops ratio: mean {:.2}, min {:.2}, max {:.2}".format(
            np.mean(ops_ratio), np.min(ops_ratio), np.max(ops_ratio)))

    return scores



class SequenceMatchScorer(object):
    def __init__(self, remove_stop_words):
        self.parser = spacy.load('en_core_web_sm', disable=['ner'])
        self.remove_stop_words = remove_stop_words

    def clean_base(self, text):
        parsed = self.parser(text)

        res = []
        for i in range(len(parsed)):
            if not self.remove_stop_words or not parsed[i].is_stop:
                res.append(parsed[i].lemma_)

        return res

    @staticmethod
    def clean_structural(text):
        return [token for token in text.split(' ') if token.startswith('@@')]

    def get_match_score(self, prediction, target, processing="base"):
        assert processing in ["base", "structural"]

        if processing == "structural":
            prediction_clean = self.clean_structural(prediction)
            target_clean = self.clean_structural(target)
            if prediction_clean == [] and target_clean == []:
                return 1.0
        else:
            prediction_clean = self.clean_base(prediction)
            target_clean = self.clean_base(target)

        sm = SequenceMatcher(a=prediction_clean, b=target_clean)

        return sm.ratio()

    def get_match_scores(self, predictions, targets, processing):
        scores = []

        num_examples = len(predictions)
        for i in tqdm(range(num_examples)):
            score = self.get_match_score(predictions[i], targets[i], processing)
            scores.append(score)

        return scores


def print_first_example_scores(evaluation_dict, num_examples):
    for i in range(num_examples):
        print("evaluating example #{}".format(i))
        print("\tsource (question): {}".format(evaluation_dict["question"][i]))
        print("\tprediction (decomposition): {}".format(evaluation_dict["prediction"][i]))
        print("\ttarget (gold): {}".format(evaluation_dict["gold"][i]))
        print("\texact match: {}".format(round(evaluation_dict["exact_match"][i], 3)))
        print("\tmatch score: {}".format(round(evaluation_dict["match"][i], 3)))
        print("\tstructural match score: {}".format(round(evaluation_dict["structural_match"][i], 3)))
        print("\tsari score: {}".format(round(evaluation_dict["sari"][i], 3)))
        print("\tGED score: {}".format(
            round(evaluation_dict["ged"][i], 3) if evaluation_dict["ged"][i] is not None
            else '-'))
        print("\tstructural GED score: {}".format(
            round(evaluation_dict["structural_ged"][i], 3) if evaluation_dict["structural_ged"][i] is not None
            else '-'))
        print("\tGED+ score: {}".format(
            round(evaluation_dict["ged_plus"][i], 3) if evaluation_dict["ged_plus"][i] is not None
            else '-'))

def print_score_stats(evaluation_dict):
    print("\noverall scores:")

    for key in evaluation_dict:
        # ignore keys that do not store scores
        if key in ["question", "gold", "prediction"]:
            continue
        score_name, scores = key, evaluation_dict[key]

        # ignore examples without a score
        if None in scores:
            scores_ = [score for score in scores if score is not None]
        else:
            scores_ = scores

        mean_score, max_score, min_score = np.mean(scores_), np.max(scores_), np.min(scores_)
        print("{} score:\tmean {:.3f}\tmax {:.3f}\tmin {:.3f}".format(
            score_name, mean_score, max_score, min_score))

def write_evaluation_output(output_path_base, num_examples, **kwargs):
    # write evaluation summary
    with open(output_path_base + '_summary.tsv', 'w') as fd:
        fd.write('\t'.join([key for key in sorted(kwargs.keys())]) + '\n')
        for i in range(num_examples):
            fd.write('\t'.join([str(kwargs[key][i]) for key in sorted(kwargs.keys())]) + '\n')

    # write evaluation scores per example
    df = pandas.DataFrame.from_dict(kwargs, orient="columns")
    df.to_csv(output_path_base + '_full.tsv', sep='\t', index=False)

def evaluate(questions, decompositions, golds, metadata,
             output_path_base, num_processes):
    decompositions_str = [d.to_string() for d in decompositions]
    golds_str = [g.to_string() for g in golds]

    # calculating exact match scores
    exact_match = [d.lower() == g.lower() for d, g in zip(decompositions_str, golds_str)]

    # evaluate using SARI
    sources = [q.split(" ") for q in questions]
    predictions = [d.split(" ") for d in decompositions_str]
    targets = [[g.split(" ")] for g in golds_str]
    sari, keep, add, deletion = get_sari(sources, predictions, targets)
    print("sari:", np.mean(sari))

    # evaluate using sequence matcher
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    match_ratio = sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                                   processing="base")
    structural_match_ratio = sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                                              processing="structural")

    # evaluate using graph distances
    graph_scorer = GraphMatchScorer()
    decomposition_graphs = [d.to_graph() for d in decompositions]
    gold_graphs = [g.to_graph() for g in golds]

    ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs)
    structural_ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs,
                                                                        structure_only=True)
    ged_plus_scores = get_ged_plus_scores(decomposition_graphs, gold_graphs,
                                          exclude_thr=5, num_processes=num_processes)

    evaluation_dict = {
        "question": questions,
        "gold": golds_str,
        "prediction": decompositions_str,
        "exact_match": exact_match,
        "match": match_ratio,
        "structural_match": structural_match_ratio,
        "sari": sari,
        "ged": ged_scores,
        "structural_ged": structural_ged_scores,
        "ged_plus": ged_plus_scores
    }
    num_examples = len(questions)
    print_first_example_scores(evaluation_dict, min(5, num_examples))
    print_score_stats(evaluation_dict)
    print("skipped {} examples when computing ged.".format(
        len([score for score in ged_scores if score is None])))
    print("skipped {} examples when computing structural ged.".format(
        len([score for score in structural_ged_scores if score is None])))
    print("skipped {} examples when computing ged plus.".format(
        len([score for score in ged_plus_scores if score is None])))

    if output_path_base:
        write_evaluation_output(output_path_base, num_examples, **evaluation_dict)

    if metadata is not None:
        metadata = metadata[metadata["question_text"].isin(evaluation_dict["question"])]
        metadata["dataset"] = metadata["question_id"].apply(lambda x: x.split("_")[0])
        metadata["num_steps"] = metadata["decomposition"].apply(lambda x: len(x.split(";")))
        score_keys = [key for key in evaluation_dict if key not in ["question", "gold", "prediction"]]
        for key in score_keys:
            metadata[key] = evaluation_dict[key]

        for agg_field in ["dataset", "num_steps"]:
            df = metadata[[agg_field] + score_keys].groupby(agg_field).agg("mean")
            print(df.round(decimals=3))

    return evaluation_dict

PATH = "C:\\Users\\Osher\\Desktop\\oren\\oren-qdmr-output\\models\\qdmr_experiment\\pred_copy.tsv"
OUT_VALUE = 'C:\\Users\\Osher\\Desktop\\oren\\evaluate.txt'

def main():
    # x = get_sari_score('flights from baltimore to san francisco'.split(' '),
    #                    'flights @@sep@@ @@1@@ from baltimore @@sep@@ @@2@@ to san francisco'.split(' '),
    #                    ['flights @@SEP@@ @@1@@ from baltimore @@SEP@@ @@2@@ to san francisco'.lower().split(' ')],
    #                    4,
    #                    0)
    # print(x)
    # x = get_sari_score('what are the stats'.split(' '), ['states'], [['states']], 4, 0)
    # print(x)
    # return

    with open(PATH, 'r') as f:
        # NOTE: TO LOWER!!
        # data = [x.lower().strip().split('\t') for x in f.readlines()]
        data = [x.strip().split('\t') for x in f.readlines()]

    questions = [d[0] for d in data]
    decompositions = [get_decomposition_from_string(d[1]) for d in data]
    golds = [get_decomposition_from_string(d[2]) for d in data]

    x = evaluate(questions, decompositions, golds, metadata=None, output_path_base=OUT_VALUE, num_processes=None)
    import pdb;
    pdb.set_trace()
    # print(x)

if __name__ == '__main__':
    main()