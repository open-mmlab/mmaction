import functools

def rsetattr(obj, attr, val):
    '''
        See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    '''
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rhasattr(obj, attr, *args):
    def _hasattr(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        else:
            return None
    return functools.reduce(_hasattr, [obj] + attr.split('.')) is not None
    
