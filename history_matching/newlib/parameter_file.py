from history_matching.newlib.quick_read import quick_read

# used for reading parameter definitions
class ParameterFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.parameters = quick_read(filename, 'Params').set_index('Name')
