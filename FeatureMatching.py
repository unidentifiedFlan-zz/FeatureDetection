import numpy as np
import sys

def calc_distance(feature1, feature2):
    return np.linalg.norm(feature1-feature2)

def nearest_neighbour(reference_descriptors, features, feature_descriptors):
    matches = {}
    threshold = 0.1
    for j, refF in enumerate(reference_descriptors):
        smallest_dist = sys.maxsize
        nearest_neighbour = -1
        for i, fd in enumerate(feature_descriptors):
            dist = calc_distance(refF, fd)
            if dist < threshold and dist < smallest_dist:
                smallest_dist = dist
                nearest_neighbour = i
        if nearest_neighbour > -1 and features:
            matches[j] = features[nearest_neighbour]
            del features[nearest_neighbour]
            del feature_descriptors[nearest_neighbour]

    return matches

def cross_nearest_neighbour(reference_features, reference_descriptors, features, feature_descriptors):
    matches = nearest_neighbour(reference_descriptors, features, feature_descriptors)
    reverse_matches = nearest_neighbour(feature_descriptors, reference_features, reference_descriptors)
    invalid_matches = []
    for key in matches.keys():
        if not reverse_matches.__contains__(key):
            invalid_matches.append(key)

    for invalid in invalid_matches:
        matches.pop(invalid)
    return matches

def nearest_neighbour_distance_ratio(reference_descriptors, features, feature_descriptors):
    matches = {}
    threshold = 1
    for j, refF in enumerate(reference_descriptors):
        smallest_dist = sys.maxsize
        second_smallest_dist = sys.maxsize
        nearest_neighbour = -1
        for i, fd in enumerate(feature_descriptors):
            dist = calc_distance(refF, fd)
            if dist < threshold and dist < smallest_dist:
                second_smallest_dist = smallest_dist
                smallest_dist = dist
                nearest_neighbour = i
            elif dist < threshold and dist < second_smallest_dist:
                second_smallest_dist = dist
        if nearest_neighbour > -1 and second_smallest_dist > 0 and smallest_dist/second_smallest_dist < 0.8:
            matches[j] = features[nearest_neighbour]
            del features[nearest_neighbour]
            del feature_descriptors[nearest_neighbour]

    return matches