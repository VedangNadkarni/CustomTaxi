import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np


# MAP = [
#     "+---------+",
#     "|R: | : :G|",
#     "| : : : : |",
#     "| : : : : |",
#     "| | : | : |",
#     "|Y| : |B: |",
#     "+---------+",
# ]

def create_map(rows=5, cols=5, locs=4, cramped=0.2, locs_prob=3, no_walls=True):

    MAP = ['+-']

    for i in range(cols-1):
        MAP[-1] = MAP[-1]+'--'

    MAP[-1] = MAP[-1]+'+'

    locs_set = [i for i in range(locs)]
    locs_set.append(' ')
    dropoffs = []
    dropoff_locs = []
    all_locs_in_prob = locs_prob/(rows*cols)
    
    p = [ all_locs_in_prob for b in range(locs)]
    p.append(1-(all_locs_in_prob*locs))

    for ro in range(rows):
        ro_str = "|"
        for co in range(cols):
            while 1:
                a = np.random.choice(locs_set, size = 1, replace=False, p=p )[0] #change this to change the randomised method of placing drop-off/pickup locations on the map
                if a not in dropoffs:
                    dropoffs.append(str(a)) if str(a) != ' ' else 1
                    dropoff_locs.append((ro,co)) if str(a) != ' ' else 1
                    if np.char.str_len(a)>1:
                        a = chr(int(a)+55)
                    ro_str = ro_str + str(a)
                    if co < cols-1:
                        a = np.random.choice(['|', ':'], size = 1, replace=True, p = [cramped, 1-cramped])[0]
                        ro_str = ro_str + str(a)
                    break

                else:
                    ro_str = ro_str + str(' ')
                    if co < cols-1:
                        a = np.random.choice(['|', ':'], size = 1, replace=True, p = [cramped, 1-cramped])[0]
                        ro_str = ro_str + str(a)
                    break
                
        ro_str = ro_str + '|'
        MAP.append(ro_str)

    MAP.append('+-')
    for i in range(cols-1):
        MAP[-1] = MAP[-1]+'--'
    
    MAP[-1] = MAP[-1]+'+'

    if no_walls:
        for co in range(cols-1):
            walls = 0
            for ro in range(rows):
                walls += 1 if MAP[ro+1][2*co+2] == '|' else 1
            if walls== rows:
                a = np.random.randint(0,rows)
                MAP[a+1] = MAP[a+1][:2*co+2]+':'+MAP[a+1][2*co+3:]
    
    return MAP, dropoff_locs



num_rows = 20
num_columns = 20
loc_no = 20
MAP, locs = create_map(num_rows,num_columns,loc_no,0.3,2,True)
loc_no = len(locs)

class CustomTaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends.

    Observations: 
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations. 
    
    Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, B and Y): locations for passengers and destinations

    actions:
    - 0: south
    - 1: north
    - 2: east
    - 3: west
    - 4: pickup
    - 5: dropoff

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        self.rows = num_rows
        self.cols = num_columns
        self.locs = locs
        self.loc_no = self.locs.__len__()
        self.num_states = self.cols*self.rows*self.loc_no*(self.loc_no+1)

        max_row = self.rows - 1
        max_col = self.cols - 1
        initial_state_distrib = np.zeros(self.num_states)
        self.num_actions = 6
        self.li = [[0, self.rows], [1, self.cols], [2, self.loc_no+1], [3, self.loc_no]]
        P = {state: {action: []
                     for action in range(self.num_actions)} for state in range(self.num_states)}
        for row in range(self.rows):
            for col in range(self.cols):
                for pass_idx in range(self.loc_no + 1):  # +1 for being inside taxi
                    for dest_idx in range(self.loc_no):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < self.loc_no and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(self.num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1 # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if (pass_idx < self.loc_no and taxi_loc == locs[pass_idx]):
                                    new_pass_idx = self.loc_no
                                else: # passenger not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == self.loc_no:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 20
                                elif (taxi_loc in locs) and pass_idx == self.loc_no:
                                    new_pass_idx = locs.index(taxi_loc)
                                else: # dropoff at wrong location
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx)
                            P[state][action].append(
                                (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, self.num_states, self.num_actions, P, initial_state_distrib)

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        self.li = [[0, self.rows, taxi_row], [1, self.cols, taxi_col], [2, self.loc_no+1, pass_loc], [3, self.loc_no, dest_idx]]
        self.li.sort(key = lambda e: e[1], reverse=True)
        i = self.li[0][2]
        for j in range(self.li.__len__()-1):
            i *= self.li[j+1][1]
            i += self.li[j+1][2]
        return i

    def decode(self, i):
        self.li = [[0, self.rows], [1, self.cols], [2, self.loc_no+1], [3, self.loc_no]]
        self.li.sort(key = lambda e: e[1], reverse=True)
        self.li.reverse()    

        for j in range(self.li.__len__()-1):
            self.li[j].append(i % self.li[j][1])
            i = i // self.li[j][1]

        self.li[-1].append(i)
        assert 0 <= i < self.li[-1][1]
        self.li.sort(key = lambda e: e[0])
        return tuple(i[-1] for i in self.li)


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        if pass_idx < self.loc_no:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
