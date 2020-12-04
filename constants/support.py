def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            wrapper.output = f(*args, **kwargs)
            return wrapper.output
        else:
            return wrapper.output

    wrapper.has_run = False
    wrapper.output = None
    return wrapper
