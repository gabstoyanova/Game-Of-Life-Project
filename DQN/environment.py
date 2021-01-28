import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

class GameOfLifeEnv():
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def __init__(self, seed=None, grid_size=5, density=0.2, save_states=False, max_generations = 400):
        
        self.grid_size = grid_size
        self.density = density
        self.save_states = save_states
        
        self.action_dict = {}
        self.max_generations = max_generations
        self.seed = seed
        self.reset()
        
    def step(self, action):
        
        aux_grid = self.state.copy()

        # take action
        if action != None:
            row = action // self.grid_size
            col = action % self.grid_size
            aux_grid[row, col] = 1
#         print('steita sq, predi evol:\n', aux_grid)

        
        # evolution by game rules
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                live = self.count_live_neighbours(aux_grid, row, col)

                # if cell is alive and has exactly 2 or 3 living neighbours, it survives
                if aux_grid[row][col] == 1 and (live == 2 or live == 3):
                    self.state[row][col] = 1

                # if cell is dead and has exactly 3 living neighbours, it lives
                elif aux_grid[row][col] == 0 and live == 3:
                    self.state[row][col] = 1
                    
                else: 
                    self.state[row][col] = 0
          

        reward = np.sum(self.state) / self.grid_size ** 2
        self.generation += 1
		
        if self.save_states:
            self.states[self.generation] = np.copy(self.state)
            
        done = False
        # print(self.generation, self.max_generations)

        if (np.sum(self.state) / self.grid_size ** 2 == 0
#                     or np.array_equal(self.state, aux_grid)
                    or self.generation > self.max_generations ):
            
            done = True
            if self.save_states:
                self.states = self.states[:self.generation + 1]

        # f = lambda x: [x]
        # grozen_state = [[f(j) for j in i] for i in self.state]
        return self.state, reward, done
        
        
    def count_live_neighbours(self, grid, row, col):
        live = 0
        n = self.grid_size
        for i in range(row - 1, row + 2, 1):
            for j in range(col - 1, col + 2, 1):
                live += grid[(i + n) % n][(j + n) % n]
        live -= grid[row][col]
        return live
    
    def reset(self):
        n = self.grid_size
        self.generation = 0
#         np.random.seed(self.seed)
        rng = np.random.RandomState(self.seed)

        # init grid 
        grid = rng.rand(self.grid_size, self.grid_size)
        for row in range(n):
            for col in range(n):
                grid[row][col] = 1 if grid[row][col] < self.density else 0
                
        self.state = grid
        
        if self.save_states:
            self.states = np.zeros((self.max_generations + 3 , n, n))
            self.states[0] = self.state
        
        f = lambda x: [x]
        grozen_state = [[f(j) for j in i] for i in self.state]
        return self.state
    
        
    def render(self):
        if not self.save_states:
            print('Render function can only be used if the flag save_states is True !')
             
        fig, ax = plt.subplots(1, 1, figsize=(4, 4)) 
        plt.grid(color='w', linestyle='-', linewidth=(8/self.grid_size))
        
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1))
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1))
        ax.set_xticklabels(['']*(self.grid_size+1))
        ax.set_yticklabels(['']*(self.grid_size+1))

        im = ax.imshow(self.states[0], vmin=0, vmax=2, cmap=plt.cm.gray)

        def init():
            im.set_data(self.states[0])
        
        def animate(i):
            im.set_data(self.states[i])
            ax.set_title('generation {} \n live: {}'.format(i, np.sum(self.states[i])))   

        ani = matplotlib.animation.FuncAnimation(fig, animate, init_func = init, frames = range(len(self.states)), interval=150)
        return ani

    def take_random_action(self):
        return int(np.random.uniform(0, self.grid_size**2, 1))