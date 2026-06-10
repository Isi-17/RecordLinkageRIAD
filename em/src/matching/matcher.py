import os
import pickle
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from src.data import full_data_utils
from src.data.dataset import DataincDataset
from src.helpers.path_helper import *
from src.models.config import Config
from src.models.pytorch_model import PyTorchModel


class Matcher(ABC):

    def __init__(self, model: PyTorchModel, processed_folder_path: str = None, results_path: str = None):
        self.model = model
        if processed_folder_path:
            self.processed_folder_path = processed_folder_path
        else:
            self.processed_folder_path = dataset_processed_folder_path(dataset_name=self.model.dataset.name)

        if results_path:
            self.results_path = results_path
        else:
            self.results_path = dataset_results_folder_path__with_subfolders(subfolder_list=[self.model.dataset.name, self.model.args.experiment_name])

    def test_records_from_positive_matches(self, test_entity_data: pd.DataFrame):
        """
        Filter the given test_entity_data down to the records that are actually
        part of the positive matches in the test set, i.e. removing all the records
        that are only part of it because they are one half of a negative match.
        """
        test_df = self.model.dataset.test_df
        test_df = test_df[test_df['label'] == 1]

        # all the unique values of 'lid' and 'rid' in test_df
        test_ids = set()
        test_ids.update(list(test_df['lid']))
        test_ids.update(list(test_df['rid']))

        test_records = test_entity_data[test_entity_data['id'].isin(test_ids)]
        return test_records

    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        
        candidates_df = self.model.dataset.test_df.copy()
        
        candidates_df = candidates_df[['lid', 'rid', 'label']]
        candidates_df['match_type'] = 'pre_serialized_pair'

        return candidates_df

    def get_test_entity_data(self) -> pd.DataFrame:
        """
        basic implementation to get test_entity_data
        can be overwritten if needed by the respective subclass
        """
        # return pd.DataFrame({'data_source_id': [1, 2]})
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        test_id_df = self.model.dataset.test_df

        full_entity_data = self.model.dataset.get_entity_data()

        test_ids = set()
        test_ids.update(list(test_id_df['lid']))
        test_ids.update(list(test_id_df['rid']))

        test_entity_data = full_entity_data[full_entity_data['id'].isin(test_ids)]
        # Save the test_entity_data
        test_entity_data.to_csv(os.path.join(self.processed_folder_path, 'test_entity_data.csv'), index=False)

        return test_entity_data

    def save_test_candidates(self, candidates_df: pd.DataFrame):
        """
        saves a blocked candidates_df with cols (lid, rid, label, match_type) in the processed path of the ds
        """
        # Delete the rows with lid == rid
        candidates_df = candidates_df[candidates_df['lid'] != candidates_df['rid']]
        # Delete duplicates
        candidates_df = candidates_df.drop_duplicates(subset=['lid', 'rid'])
        # Save the test candidates
        candidates_df.to_csv(os.path.join(self.processed_folder_path, 'full_test_candidates.csv'), index=False)

    def pre_cleanup(self, pairwise_matches_preds: pd.DataFrame, threshold: float = 0.999, num_of_datasources: int = 5):
        """
        Pre cleanup step before running the graph cleanup. In the base class, we just return the matches_graph.
        In subclasses we addtionally break up large subgraphs that make the graph cleanup run too long.
        """
        matches_graph = full_data_utils.generate_matches_graph(pairwise_matches_preds, threshold=threshold)

        _, transitive_matches = full_data_utils.generate_transitive_matches_graph(matches_graph,
                                                                                  add_transitive_edges=False,
                                                                                  results_path=self.results_path,
                                                                                  subgraph_size_threshold=100)

        # Save the pre-cleanup transitive matches
        transitive_matches_df = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
        transitive_matches_df.to_csv(os.path.join(self.results_path, 'pre_cleanup_transitive_matches.csv'), index=False)

        return matches_graph

    def graph_cleanup(self, matches_graph, num_of_datasources=5):
        """
        Clean up the matches graph by breaking up large subgraphs and removing edges with high betweenness centrality
        :param num_of_datasources: The number of data sources in the dataset
        :return: files of the cleaned up matches graph and the deleted edges
        """

        print('=' * 50)
        print('Starting graph cleanup')
        print('=' * 50)
        deleted_edges_dict = {'lid': [], 'rid': [], 'prob': [], 'match_type': []}

        ###############################################################################################################

        # 1st Cleanup: Break up subgraphs bigger than 5*num_of_datasources via minimum edge cuts.

        ###############################################################################################################

        print('=' * 50)
        print('Starting minimum edge cut cleanup')
        print('=' * 50)

        # While there are subgraphs with more than 5*number_of_datasources nodes, we break them up via minimum edge cuts
        subgraphs = list(nx.connected_components(matches_graph))

        while any([len(c) > 3 * num_of_datasources for c in subgraphs]):
            largest_subgraph = max(subgraphs, key=len)
            print('Largest subgraph size: {}, Number of subgraphs: {}'.format(len(largest_subgraph),
                                                                              len(subgraphs)))
            # Gather all subgraphs with the maximum size
            largest_subgraphs = [c for c in subgraphs if len(c) == len(largest_subgraph)]

            # Clean up the largest subgraphs via minimum edge cuts
            for subgraph_idx, c in enumerate(largest_subgraphs):

                matches_graph, deleted_edges_dict = self.minimum_edge_cut_clean_up(c, matches_graph, deleted_edges_dict)

            subgraphs = list(nx.connected_components(matches_graph))

        ###############################################################################################################

        # 2nd Cleanup: Remove the edges with the highest betweenness centrality in each subgraph with more than
        # num_of_datasources nodes.

        ###############################################################################################################
        print('=' * 50)
        print('Starting betweenness centrality cleanup')
        print('=' * 50)

        subgraphs = list(nx.connected_components(matches_graph))

        while any([len(c) > num_of_datasources for c in subgraphs]):
            # Get the subgraphs with more than num_of_datasources nodes
            large_subgraphs = [c for c in subgraphs if len(c) > num_of_datasources]
            print('Number of subgraphs with more than {} nodes: {}, Number of subgraphs: {}'.format(num_of_datasources,
                                                                                                    len(large_subgraphs),
                                                                                                    len(subgraphs)))

            # Clean up the large subgraphs via removing the edge with the highest betweenness centrality
            for subgraph_idx, c in enumerate(large_subgraphs):
                # Compute the betweenness centrality of all edges of the subgraph
                subgraph = matches_graph.subgraph(large_subgraphs[subgraph_idx])
                betweenness_centrality = nx.edge_betweenness_centrality(subgraph)
                # Get the edge with the highest betweenness centrality
                max_betweenness_edge = max(betweenness_centrality, key=betweenness_centrality.get)
                # Record the deleted edge on the deleted_edges_dict
                deleted_edges_dict['lid'].append(max_betweenness_edge[0])
                deleted_edges_dict['rid'].append(max_betweenness_edge[1])
                deleted_edges_dict['prob'].append(matches_graph[max_betweenness_edge[0]][max_betweenness_edge[1]]['prob'])
                deleted_edges_dict['match_type'].append(matches_graph[max_betweenness_edge[0]][max_betweenness_edge[1]]['match_type'])
                # Remove the edge with the highest betweenness centrality from the matches_graph
                matches_graph.remove_edge(max_betweenness_edge[0], max_betweenness_edge[1])

            subgraphs = list(nx.connected_components(matches_graph))

        ###############################################################################################################

        # Final Graph Step: Add all the transitive edges of each subgraph to the matches_graph

        ###############################################################################################################

        matches_graph, _ = full_data_utils.generate_transitive_matches_graph(matches_graph, True)

        # Save the edges of the post graph cleanup matches_graph, with their match_type

        matches_graph_df = pd.DataFrame(matches_graph.edges(data=True), columns=['lid', 'rid', 'match_type'])
        matches_graph_df.to_csv(os.path.join(self.results_path, 'post_graph_cleanup_matches.csv'), index=False)

        # Save the deleted edges

        deleted_edges_df = pd.DataFrame(deleted_edges_dict)
        deleted_edges_df.to_csv(os.path.join(self.results_path, 'graph_cleanup_deleted_edges.csv'), index=False)

        print('Finished graph cleanup')

    ###########################################################################

    # Utils

    ###########################################################################

    def minimum_edge_cut_clean_up(self, subgraph, matches_graph, deleted_edges_dict):

        # Get the minimum edge cut of the subgraph
        subgraph = matches_graph.subgraph(subgraph)
        min_edge_cut = nx.minimum_edge_cut(subgraph)

        # Save the deleted edges with their lid, rid, prob attributes
        deleted_edges_lids = [lid for lid, rid in min_edge_cut]
        deleted_edges_rids = [rid for lid, rid in min_edge_cut]
        deleted_edges_probs = [matches_graph[lid][rid]['prob'] for lid, rid in min_edge_cut]
        deleted_edges_match_types = [matches_graph[lid][rid]['match_type'] for lid, rid in min_edge_cut]
        deleted_edges_dict['lid'].extend(deleted_edges_lids)
        deleted_edges_dict['rid'].extend(deleted_edges_rids)
        deleted_edges_dict['prob'].extend(deleted_edges_probs)
        deleted_edges_dict['match_type'].extend(deleted_edges_match_types)

        # Remove the cut edges from the graph
        matches_graph.remove_edges_from(min_edge_cut)

        return matches_graph, deleted_edges_dict


    def run_matching(self, args):
        """
        Runs the whole matching pipeline:
        - A) blocking
        - B) pairwise matching
        - C) graph cleanup
        """

        test_entity_data = self.get_test_entity_data()

        print(test_entity_data)
        

        # A) get candidates using the blocking function
        candidate_df = self.blocking(test_entity_data)


        print("Total candidates:", len(candidate_df))
        print("Duplicates:", candidate_df.duplicated(subset=['lid', 'rid']).sum())
        print("Self matches:", (candidate_df['lid'] == candidate_df['rid']).sum())
        
        # drop match_type for now to use the testing function
        candidate_idx_df = candidate_df.drop(columns=['match_type'])
        
        # inject the candidates into the test_data_loader, not the best way, but quick for now
        self.model.test_data_loader.dataset.idx_df = candidate_idx_df

        # B) run pairwise matching

        # First check if the pairwise_matches_preds have already been previously saved
        pairwise_matches_preds_path = os.path.join(self.results_path, 'pairwise_matches_preds.csv')

        if file_exists_or_create(pairwise_matches_preds_path):
            self.pairwise_matches_preds = pd.read_csv(pairwise_matches_preds_path)
        else:
            # Run the pairwise matching
            self.model.test(epoch=args.epoch) 
            self.pairwise_matches_preds = self.load_and_save_pairwise_matches_preds(args, candidate_df)

        # C) perform graph_cleanup
        # if 'data_source_id' in test_entity_data.columns:
        #     num_ds = test_entity_data['data_source_id'].nunique()
        #     print("Num ds: ", num_ds)
        # else:
        #     # Get the raw_df
        #     raw_df = self.model.dataset.get_raw_df()
        #     if 'data_source_id' in raw_df.columns:
        #         num_ds = raw_df['data_source_id'].nunique()
        #     else:
                # If the dataset doesn't have a set number of data sources, we set num_ds arbitrarily to 5.
                # num_ds = 5
        num_ds = 5
        
        #Check if threshold is set in args
        if hasattr(args, 'threshold'):
            self.matches_graph = self.pre_cleanup(self.pairwise_matches_preds, threshold=args.threshold, num_of_datasources=num_ds)
        else:
            self.matches_graph = self.pre_cleanup(self.pairwise_matches_preds, threshold=0.999, num_of_datasources=num_ds)

        self.graph_cleanup(self.matches_graph, num_of_datasources=num_ds)

    def load_and_save_pairwise_matches_preds(self, args, candidate_df):
        """
        Loads the pairwise_matches_preds from the prediction_log and saves them to the processed folder
        """
        file_name = "".join([self.model.args.model_name, '__prediction_log__ep', str(args.epoch), '.csv'])
        log_path = experiment_file_path(args.experiment_name, file_name)

        pairwise_matches_preds = pd.read_csv(log_path)
        
        # Add the match_type column to the pairwise_matches_preds
        # pairwise_matches_preds = pairwise_matches_preds.merge(candidate_df[['lid', 'rid', 'match_type']], left_on=['lids', 'rids'], right_on=['lid', 'rid'])
        unique_candidates = candidate_df[['lid', 'rid', 'match_type']].drop_duplicates(subset=['lid', 'rid'])
        
        pairwise_matches_preds = pairwise_matches_preds.merge(
            unique_candidates, 
            left_on=['lids', 'rids'], 
            right_on=['lid', 'rid'],
            how='left'
        )
        
        pairwise_matches_preds = pairwise_matches_preds.drop(columns=['labels', 'predictions', 'lid', 'rid'])
        # Rename the column 'prediction_proba' to 'prob'
        pairwise_matches_preds = pairwise_matches_preds.rename(columns={'prediction_proba': 'prob'})
        # Rename the lids and rids columns to lid and rid
        pairwise_matches_preds = pairwise_matches_preds.rename(columns={'lids': 'lid', 'rids': 'rid'})

        # Save the pairwise_matches_preds
        pairwise_matches_preds.to_csv(os.path.join(self.results_path, 'pairwise_matches_preds.csv'), index=False)

        return pairwise_matches_preds

    ###########################################################################

    # Blocking Utils

    ###########################################################################

    def get_tknzd_records_and_overlap_indicators(self, test_entity_data):
        tokenized_records = self.model.dataset.get_tokenized_data()
        # The tokenized records are indexed by the id of the raw data
        tokenized_test_records = tokenized_records[tokenized_records.index.isin(test_entity_data['id'])]
        # Generate the list of all tokens seen in the test records
        tmp_list = tokenized_test_records['tokenized'].apply(lambda x: list(set(x)))
        tmp_list = tmp_list.apply(lambda x: list(set(x)))
        all_tokens = np.array(list(set(string for sublist in tmp_list for string in sublist)))
        # Index structure for much faster lookup of the index positions of every token in the all_tokens list
        # (so that we do not have to call all_tokens.index but rather get a O(1) lookup)
        index_lookup = {value: i for i, value in enumerate(all_tokens)}
        # Generating a sparse matrix with (n_records, n_tokens), where a 1 at (recordX, tokenY) indicates,
        # that recordX contains the tokenY
        #
        data = []
        row = []
        col = []
        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()),
                                                        total=tokenized_test_records.shape[0],
                                                        desc='Building indices matrix'):
            token_indexes_in_record = sorted(set([index_lookup[t] for t in tokenized_record['tokenized']]))

            n_tokens = len(token_indexes_in_record)
            data.extend([True for _ in range(n_tokens)])
            row.extend([i for _ in range(n_tokens)])
            col.extend(token_indexes_in_record)
        indicators = csr_matrix((data, (row, col)), shape=(tokenized_test_records.shape[0], len(all_tokens)),
                                dtype=np.int8)
        return indicators, tokenized_test_records

    def get_top_overlap_idx(self, i, indicators, test_entity_data):
            lookup = np.array(indicators[i, :].dot(indicators.transpose()).todense())[0]
            # Set all records from the same data source to zero, because we only want matches with other data sources
            current_data_source = test_entity_data.iloc[i]['data_source_id']
            multiplication_mask = np.ones(test_entity_data.shape[0], dtype=np.int8) - \
                np.array(test_entity_data['data_source_id'] == current_data_source)
            lookup *= multiplication_mask
            top_overlap_idx = np.argpartition(lookup, -self.number_of_candidates)[-self.number_of_candidates:]
            return top_overlap_idx

    def get_top_overlap_idx_one_source(self, i, indicators, test_entity_data):
            lookup = np.array(indicators[i, :].dot(indicators.transpose()).todense())[0]
            top_overlap_idx = np.argpartition(lookup, -self.number_of_candidates)[-self.number_of_candidates:]
            return top_overlap_idx

