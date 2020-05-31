import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np



def plot_blackjack_values(V):
    '''
    Credits : https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/plot_utils.py
    '''
    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(12, 22)
        y_range = np.arange(1, 10)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()

def plot_blackjack_policy(policy):
    '''
    Credits : https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/plot_utils.py
    '''
    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in policy:
            return policy[x,y,usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x,y,usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0,1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
            
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()

def plot_avg_rewards_cliff_walking(rewards_dict : dict) -> None:
    fig = plt.figure()
    for agent,reward in rewards_dict.items():
        reward = np.array(reward)
        num_episodes = reward.shape[1]
        plt.plot(range(1,num_episodes+1), np.mean(reward, axis = 0), label = agent)
    plt.xlabel('Episodes')
    plt.ylabel("Sum of rewards during episode")
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()
    
def plot_avg_rewards(rewards_dict : dict) -> None:
    fig = plt.figure()
    for agent,reward in rewards_dict.items():
        reward = np.array(reward)
        max_num_steps = reward.shape[1]
        plt.plot(range(1,max_num_steps+1), np.mean(reward, axis = 0), label = agent)
    plt.xlabel('Steps')
    plt.ylabel("Avg. rewards")
    plt.legend()
    plt.show()

def plot_optimal_actions(optimal_actions_dict : dict) -> None:
    fig = plt.figure()
    for agent,optimal_actions in optimal_actions_dict.items():
        optimal_actions = np.array(optimal_actions)
        max_num_steps = optimal_actions.shape[1]
        plt.plot(range(1,max_num_steps+1) , np.mean(optimal_actions, axis = 0) * 100, label = agent)
    plt.xlabel('Steps')
    plt.ylabel("% Optimal Action")
    plt.legend()
    plt.show()