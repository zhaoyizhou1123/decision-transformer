'''
utils for point maze
'''
from copy import deepcopy

VALID_VALUE = [0,1,'c','g','r'] # valid value of cell

def set_map_cell(MAP, pos, value):
    '''
    Return a map with cell [pos] of MAP set to [value], MAP is not changed.
    MAP: list(list), the basic map
    pos: np.array (2,), type=int, the (row,col) of the cell. Upperleft cell is (0,0)
    value: {0,1,'c','g','r'}, the value to set the cell

    Output: list(list), a modified map
    '''
    assert value in VALID_VALUE, f"_set_map_cell: Invalid value {value}!"
    new_map = deepcopy(MAP)
    new_map[pos[0]][pos[1]] = value

    return new_map