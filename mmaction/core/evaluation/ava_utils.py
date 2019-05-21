import csv
import logging
import time
from collections import defaultdict
import heapq


def det2csv(dataset, results):
    csv_results = []
    for idx in range(len(dataset)):
        video_id = dataset.video_infos[idx]['video_id']
        timestamp = dataset.video_infos[idx]['timestamp']
        width = dataset.video_infos[idx]['width']
        height = dataset.video_infos[idx]['height']
        result = results[idx]
        for label in range(len(result)):
            for bbox in result[label]:
                _bbox = bbox.tolist()
                x1 = _bbox[0] / width
                y1 = _bbox[1] / height
                x2 = _bbox[2] / width
                y2 = _bbox[3] / height
                csv_results.append((video_id, timestamp, x1, y1, x2, y2, label + 1, _bbox[4]))
    return csv_results


def results2csv(dataset, results, out_file):
    if isinstance(results[0], list):
        csv_results = det2csv(dataset, results)
    # TODO: integrate CSVWriter into mmcv.fileio
    # mmcv.dump(csv_results, out_file)
    with open(out_file, 'w') as f:
        for csv_result in csv_results:
            f.write(','.join(map(lambda x: str(x), csv_result)))
            f.write('\n')


def print_time(message, start):
    logging.info("==> %g seconds to %s", time.time() - start, message)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def read_csv(csv_file, class_whitelist=None, capacity=0):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
        csv_file: A file object.
        class_whitelist: If provided, boxes corresponding to (integer) class labels
        not in this set are skipped.
        capacity: Maximum number of labeled boxes allowed for each example.
        Default is 0 where there is no limit.

    Returns:
        boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
        labels: A dictionary mapping each unique image key (string) to a list of
        integer class lables, matching the corresponding box in `boxes`.
        scores: A dictionary mapping each unique image key (string) to a list of
        score values lables, matching the corresponding label in `labels`. If
        scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], "Wrong number of columns: " + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue
            score = 1.0
        if len(row) == 8:
            score = float(row[7])
        if capacity < 1 or len(entries[image_key]) < capacity:
            heapq.heappush(entries[image_key],
                           (score, action_id, y1, x1, y2, x2))
        elif score > entries[image_key][0][0]:
            heapq.heapreplace(entries[image_key],
                              (score, action_id, y1, x1, y2, x2))
    for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: -tup[0])
        for item in entry:
            score, action_id, y1, x1, y2, x2 = item
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    print_time("read file " + csv_file.name, start)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
        exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
        A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
        or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
    for row in reader:
        assert len(row) == 2, "Expected only 2 columns, got: " + row
        excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
        labelmap_file: A file object containing a label map protocol buffer.

    Returns:
        labelmap: The label map in the form used by the object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
        class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    for line in labelmap_file:
        if line.startswith("  name:"):
            name = line.split('"')[1]
        elif line.startswith("  id:") or line.startswith("  label_id:"):
            class_id = int(line.strip().split(" ")[-1])
            labelmap.append({"id": class_id, "name": name})
            class_ids.add(class_id)
    return labelmap, class_ids
