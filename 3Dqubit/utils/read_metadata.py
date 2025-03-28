def read_metadata(file):
    file = open(file, 'r')
    lines = file.read().strip().split("\n")
    file.close()

    dictionary = {}
    for l in lines:
        terms = list(map(lambda s: s.strip(), l.split(",")))
        if len(terms) != 2: raise Exception("Only two terms per line in a metadata file")
        dictionary[terms[0]] = terms[1]

    return dictionary