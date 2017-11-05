import json
from reversi.game_core import Field


class GameHistory:
    def __init__(self):
        self.events = []
        self.metadata = dict()

    def decode(self, string):
        loaded = json.loads(string)
        self.events = loaded['events']
        self.metadata = loaded['metadata']

    def encode(self):
        return json.dumps({
            'events': self.events,
            'metadata': self.metadata
        }, indent=4)

    def write(self, filename):
        with open(filename, 'w') as file:
            file.write(self.encode())

    def read(self, filename):
        with open(filename, 'r') as file:
            return self.decode(file.read())

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def add_move(self, player, pos, choice):
        (x, y) = pos
        self.events.append({
            'type': 'move',
            'move': {
                'player': player.value,
                'x': x,
                'y': y,
                'c': choice
            }
        })

    def add_disqualification(self, player):
        self.events.append({
            'type': 'disqualification',
            'player': player.value
        })

    def __getitem__(self, arg):
        item = self.events[arg]

        if item['type'] == 'disqualification':
            return Disqualification(Field(item['player']))
        if item['type'] == 'move':
            pos = (item['move']['x'], item['move']['y'])
            choice = item['move']['c']
            player = Field(item['move']['player'])

            return Move(player, pos, choice)


class Disqualification:
    def __init__(self, player):
        self.player = player


class Move:
    def __init__(self, player, pos, choice):
        self.player = player
        self.pos = pos
        self.choice = choice