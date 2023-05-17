class media:
    def __init__(self, id, name):
        self.id = id
        self.name = name
    
    def __str__(self):
        return ', '.join(f"{item[0]}: {item[1]}" for item in vars(self).items())