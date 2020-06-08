import sys
from torchreid import metrics
import torch
import numpy as np

class Matching(object):

    @staticmethod
    def get_matching_candidates(track, candidates_list, interruption_threshold=5):
        candidates = []
        min_len = sys.maxsize
        for candidate in candidates_list:
            time_offset = track.start_time - candidate.end_time
            print("Time offset between {} and {}: {}".format(track.track_id, candidate.track_id, time_offset))
            if not candidate.is_completed_track() and 0 <= time_offset and time_offset <= interruption_threshold:
                candidates.append(candidate)
                if candidate.get_embeddings().size()[0] < min_len:
                    min_len = candidate.get_embeddings().size()[0]
        return candidates, min_len

    @staticmethod
    def match(matching_track, enter_tracks, interruption_threshold=5, dist_metric='cosine'):
        print("Start matching for track {}...".format(matching_track.track_id))
        candidates, min_len = Matching.get_matching_candidates(matching_track, enter_tracks.values(), interruption_threshold)
        q_embeddings = matching_track.get_embeddings().cpu()
        g_track_maps = []
        g_embeddings = []
        for can in candidates:
            embs = can.get_embeddings(min_len)
            g_embeddings.append(embs.cpu())
            g_track_maps += [can.track_id] * len(embs)

        if len(g_embeddings) >= 1:
            g_embeddings = torch.cat(g_embeddings, 0)
            distances = metrics.compute_distance_matrix(q_embeddings, g_embeddings, dist_metric)
            results = {}
            for r in distances:
                r_list = r.tolist()
                min_dist = min(r)
                min_index = r_list.index(min_dist)
                track_id = g_track_maps[min_index]
                result_data = results.get(track_id, None)
                if result_data is None:
                    results[track_id] = {'total': 1, 'min_dist': min_dist, 'track_id': track_id}
                else:
                    result_data['total'] += 1
                    if min_dist < result_data['min_dist']:
                        result_data['min_dist'] = min_dist
            final_result = {'total': 0, 'min_dist': 1000, "track_id": ""}
            for res in results.values():
                if res['total'] > final_result['total'] or (res['total'] == final_result['total'] and res['min_dist'] < final_result['min_dist']):
                    final_result = res

            print('Major voting result: ', final_result, results)
            return final_result['track_id']
        elif len(g_embeddings) == 1:
            return candidates[0].track_id

    @staticmethod
    def match_by_timestamp(tracks, interruption_threshold=5):
        matched_tracks = {}
        while len(tracks) > 0:
            track = tracks[0]
            to_del = [0]
            concated_id = []
            print('Start matching for track {}...'.format(track.track_id))
            for j in range(1, len(tracks)):
                if (not tracks[j].is_enter_track()) and (track.end_time < tracks[j].start_time) \
                        and (tracks[j].start_time - track.end_time < interruption_threshold):
                    print('>>> Candidate track {} has distance time: {}'.format(tracks[j].track_id, tracks[j].start_time - track.end_time))
                    track.concat_track(tracks[j])
                    to_del.append(j)
                    concated_id.append(tracks[j].track_id)

            matched_tracks[track.track_id] = concated_id
            # if track.is_completed_track:
            #     track_manager.remove_track(track.track_id)
            for t in reversed(to_del): tracks.pop(t)

        return matched_tracks

    @staticmethod
    def match_by_distance(tracks, interruption_threshold=5):
        matched_tracks = {}
        while len(tracks) > 0:
            track = tracks[0]
            to_del = [0]
            concated_id = []
            print('Start matching for track {}...'.format(track.track_id))
            tail_tracks = sorted(range(1, len(tracks)), key=lambda t: tracks[t].start_time)

            while True:
                min_dist = sys.maxsize
                candidate = None
                for j in tail_tracks:
                    if (not tracks[j].is_enter_track()) and (track.end_time < tracks[j].start_time) \
                            and (tracks[j].start_time - track.end_time < interruption_threshold):
                        print('>>> Candidate track {} has distance time: {}'.format(tracks[j].track_id,
                                                                                    tracks[j].start_time - track.end_time))
                        dist = np.linalg.norm(tracks[j].start_point - track.end_point)
                        print('>>> Candidate track {} has distance pixel: {}'.format(tracks[j].track_id, dist))
                        if dist < min_dist:
                            min_dist = dist
                            candidate = j
                if candidate is None: break
                track.concat_track(tracks[candidate])
                to_del.append(candidate)
                concated_id.append(tracks[candidate].track_id)

            matched_tracks[track.track_id] = concated_id
            for t in reversed(to_del): tracks.pop(t)

        return matched_tracks

    @staticmethod
    def match_by_visual(tracks, interruption_threshold=5, dist_metric='cosine'):
        matched_tracks = {}
        while len(tracks) > 0:
            track = tracks[0]
            q_embeddings = track.get_embeddings(20).cpu()
            # q_embeddings = track.get_median_embedding().cpu()
            to_del = [0]
            concated_id = []
            print('Start matching for track {}...'.format(track.track_id))
            while True:
                # Select candidates
                index_can = []
                candidates = []
                min_len = sys.maxsize
                for j in range(1, len(tracks)):
                    if (not tracks[j].is_enter_track()) and (track.end_time < tracks[j].start_time) \
                            and (tracks[j].start_time - track.end_time < interruption_threshold):
                        print('>>> Candidate track {} has distance time: {}'.format(tracks[j].track_id,
                                                                                    tracks[j].start_time - track.end_time))
                        candidates.append(tracks[j])
                        index_can.append(j)
                        # num_embs = tracks[j].get_median_embedding().size()[0]
                        num_embs = tracks[j].get_embeddings().size()[0]
                        print(num_embs)
                        if num_embs < min_len:
                            min_len = num_embs
                if len(candidates) == 0: break
                print('Min len gallery for each track: {}'.format(min_len))
                g_track_maps = []
                index_track_maps = []
                g_embeddings = []
                for t, can in enumerate(candidates):
                    embs = can.get_embeddings(min_len)
                    print(embs.size())
                    # embs = can.get_median_embedding()
                    g_embeddings.append(embs.cpu())
                    g_track_maps += [can.track_id] * len(embs)
                    index_track_maps += [index_can[t]] * len(embs)
                # print(g_track_maps)

                # Major Voting
                if len(g_embeddings) > 1:
                    g_embeddings = torch.cat(g_embeddings, 0)
                    print('Size of gallery embedding: {}'.format(g_embeddings.size()))
                    distances = metrics.compute_distance_matrix(q_embeddings, g_embeddings, dist_metric)
                    print(distances.size())
                    results = {}
                    for r in distances:
                        r_list = r.tolist()
                        min_dist = min(r)
                        min_index = r_list.index(min_dist)
                        track_id = g_track_maps[min_index]
                        index_track = index_track_maps[min_index]
                        result_data = results.get(track_id, None)
                        if result_data is None:
                            results[track_id] = {'total': 1, 'min_dist': min_dist, 'track_id': track_id, 'index': index_track}
                        else:
                            result_data['total'] += 1
                            if min_dist < result_data['min_dist']:
                                result_data['min_dist'] = min_dist
                    final_result = {'total': 0, 'min_dist': 1000, 'track_id': None, 'index': None}
                    for res in results.values():
                        if res['total'] > final_result['total'] or (
                                res['total'] == final_result['total'] and res['min_dist'] < final_result['min_dist']):
                            final_result = res
                    print('Major voting result: ', final_result, results)
                    if final_result['track_id'] is not None:
                        concated_id.append(final_result['track_id'])
                        to_del.append(final_result['index'])
                        track.concat_track(tracks[final_result['index']])

                elif len(g_embeddings) == 1:
                    concated_id.append(candidates[0].track_id)
                    to_del.append(index_can[0])
                    track.concat_track(tracks[index_can[0]])

            matched_tracks[track.track_id] = concated_id
            for t in reversed(to_del): tracks.pop(t)

        return matched_tracks