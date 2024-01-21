# https://stackoverflow.com/a/54092401/852795


def append_history(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = dict()

        for key, value in h1.items():
            dest[key] = value + h2[key]

        return dest
