import numpy as np
import os
import random
import mdtraj as mdt
import argparse


def pad_labels(labels):
    '''Converts a .labels_ attribute to a n x m 2d numpy array, in which n is the number of trajectories and m is the length of the longest trajectory in the data. Shorter trajectories are padded with -1 s.
    The reason for this is so np.where can be used later to pull frames for each MSM state (Can this be made the norm again???).'''
    trj_lengths = []
    for traj in labels:
        trj_lengths.append(len(traj))
    max_length = np.max(trj_lengths)
    output = np.concatenate((labels[0], -1 * np.ones(max_length - len(labels[0]))))
    for i in labels[1:]:
        append = np.concatenate((i, -1 * np.ones(max_length - len(i))))
        output = np.vstack((output, append))
    return output
 

def compute_centroid(frames):
    '''Computes which frames out of all pulled frames for each MSM state is the centroid.'''
    centroid_scores = []
    for frame in frames:
        centroid_scores.append(np.sum((np.sum((frames - frame) ** 2, axis=1) / len(frame)) ** 0.5))
    return np.where(centroid_scores == np.min(centroid_scores))[0][0]


def make_gens_file(trj_paths, featurized_trjs, topology, stride, padded_labels, n_samples, states):
    # Compute centroid from featurized trajectories
    if states == '-1':
        states = range(0, int(np.amax(padded_labels)) + 1)
    for i in states:
        choices = np.where(padded_labels == i)
        for j in range(n_samples):
            select = random.choice(range(len(choices)))
            choice = [choices[0][select], choices[1][select]]
            state_frame = featurized_trjs[choice[0]][choice[1]]
            if j == 0:
                state_frames = np.array([state_frame])
                frame_positions = np.array([choice])
            else:
                state_frames = np.vstack((state_frames, [state_frame]))
                frame_positions = np.vstack((frame_positions, [choice]))
        centroid_position = compute_centroid(state_frames)
        if i == 0:
            centroid_positions = np.array([frame_positions[centroid_position]])
        else:
            centroid_positions = np.vstack((centroid_positions, [frame_positions[centroid_position]]))
    # Pull from raw data
    for i in range(len(centroid_positions)):
        if topology != '-1':
            raw_frame = mdt.load(trj_paths[centroid_positions[i][0]], top=topology)
        else:
            raw_frame = mdt.load(trj_paths[centroid_positions[i][0]])
        raw_frame = raw_frame.slice(range(0, raw_frame.n_frames, stride)).slice(centroid_positions[i][1])
        if i == 0:
            output = raw_frame
        else:
            output = output.join(raw_frame)
    output = output.superpose(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Computes a representative structure for each state in an MSM and returns the structures, as a single xtc file. Used to get a visual interpretation of the MSM states, reminiscent of the "Gens" file in MSMBuilder 2.''')
    parser.add_argument('-t', '--trj_paths', type=str, help='''A text file with raw trajectory paths, the order they were loaded while you were building the MSM.''')
    parser.add_argument('-f', '--featurized_trjs', type=str, help='''Featurized trajectories file in .npy format.''')
    parser.add_argument('-s', '--stride', type=str, help='''If you strided your data before featurizing it to build your MSM, use that corresopnding stride here.''', default='1')
    parser.add_argument('-T', '--topology', type=str, help='''Topology file for loading the raw data, if needed.''', default='-1')
    parser.add_argument('-l', '--labels', type=str, help='''"labels_" attribute from a featurization object, e.g. KMeans.labels_, saved as a npy file.''')
    parser.add_argument('-n', '--n_samples', type=str, help='''How many frames from each state do you want to pull from the raw data to compute the representative "centroid" structure, which is defined as the one with the minimum total RMSD to all other structures.''', default='-1')
    parser.add_argument('-S', '--states' type=str, help='''Pull structures from a select set of states (e.g. only those surviving after ergodic trimming)
    This should be a single column text file of state indices.''', default='-1')
    parser.add_argument('-o', '--output', type=str, help='''Name of the output structures file.''')
    args = parser.parse_args()
    
    trj_paths = open(args.trj_paths).read().splitlines()
    padded_labels = pad_labels(np.load(args.labels))
    featurized_trjs = np.load(args.featurized_trjs)
    if args.states != '-1':
        args.states = np.loadtxt(args.states)
    centroids = make_gens_file(trj_paths, featurized_trjs, args.topology, int(args.stride), padded_labels, int(args.n_samples), args.states)   
    centroids.save_xtc(args.output)

