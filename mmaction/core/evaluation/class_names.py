import mmcv


def ava_classes():
    return [
        'bend/bow (at the waist)', 'crawl', 'crouch/kneel', 'dance',
        'fall down', 'get up', 'jump/leap', 'lie/sleep', 'martial art',
        'run/jog', 'sit', 'stand', 'swim', 'walk', 'answer phone',
        'brush teeth', 'carry/hold (an object)', 'catch (an object)', 'chop',
        'climb (e.g., a mountain)',
        'clink glass', 'close (e.g., a door, a box)', 'cook', 'cut', 'dig',
        'dress/put on clothing', 'drink', 'driving (e.g., a car, a truck)',
        'eat', 'enter', 'exit', 'extract', 'fishing', 'hit (an object)',
        'kick (an object)', 'lift/pick up', 'listen (e.g., to music)',
        'open (e.g., a window, a car door)', 'paint', 'play board game',
        'play musical instrument', 'play with pets', 'point to (an object)',
        'press', 'pull (an object)', 'push (an object)', 'put down', 'read',
        'ride (e.g., a bike, a car, a horse)', 'row boat', 'sail boat',
        'shoot', 'shovel', 'smoke', 'stir', 'take a photo',
        'text on/look at a cellphone', 'throw', 'touch (an object)',
        ' (e.g., a screwdriver)', 'watch (e.g., TV)', 'work on a computer',
        'write', 'fight/hit (a person)',
        'give/serve (an object) to (a person)',
        'grab (a person)', 'hand clap', 'hand shake', 'hand wave',
        'hug (a person)',
        'kick (a person)', 'kiss (a person)', 'lift (a person)',
        'listen to (a person)', 'play with kids', 'push (another person)',
        'sing to (e.g., self, a person, a group)',
        'take (an object) from (a person)',
        'talk to (e.g., self, a person, a group)', 'watch (a person)'
    ]


dataset_aliases = {
    'ava': ['ava', 'ava2.1', 'ava2.2'],
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels
