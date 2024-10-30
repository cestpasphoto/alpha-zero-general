import importlib

class_names_dict = {
    'azul': 'Azul',
    'botanik': 'Botanik',
    'minivilles': 'Minivilles',
    'santorini': 'Santorini',
    'smallworld': 'Smallworld',
    'splendor': 'Splendor',
    'thelittleprince': 'TLP'
}

def import_game(pkg_name):
    class_name = class_names_dict.get(pkg_name)
    if class_name is None:
        raise Exception(f'Game {pkg_name} is not known, please chose a game among the following list: {list(class_names_dict.keys())}')
    game_name = class_name + 'Game'
    game_module = importlib.import_module(pkg_name + '.' + game_name)
    game_class = getattr(game_module, game_name)
    nnet_module = importlib.import_module(pkg_name + '.NNet')
    nnet_class = getattr(nnet_module, 'NNetWrapper')
    players_module = importlib.import_module(pkg_name + '.' + class_name + 'Players')
    return game_class, nnet_class, players_module, getattr(game_module, "NUMBER_PLAYERS")

def import_logicnumba(pkg_name):
    mdl = importlib.import_module(pkg_name + '.' + class_names_dict[pkg_name] + 'LogicNumba')
    #if "__all__" in mdl.__dict__:
    #    names = mdl.__dict__["__all__"]
    #else:
    #    names = [x for x in mdl.__dict__ if not x.startswith("_")]
    #    globals().update({k: getattr(mdl, k) for k in names})
    board_class = getattr(mdl, 'Board')
    return board_class

