# Imports
import tensorflow as tf
from tensorforce.agents import PPOAgent, VPGAgent
from tensorforce.contrib.openai_gym import OpenAIGym
import gym
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np

# Globals
global rewards
global agg_sum
rewards = []
agg_rewards = []
agg_sum = 0 # aggregated reward
durations = []
MAXSTEPS = 50001
learning_rate = 0.001
def plotter(timestep,model_name,title=None,save_step=1000):
    f = plt.figure(figsize=(10, 10))
    if title is None:
        pass
    else:
        plt.title(title)

    plt.subplot (2,1,1)
    plt.title('Rewards over Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.grid()
    # plt.plot(range(timestep),rewards,'ro',label='Step Reward',linewidth=0.5,markersize=5)
    plt.plot(range(timestep),agg_rewards,'b--',label='Average Rewards',markersize=1.0)
    plt.legend()

    plt.subplot(2,1,2)
    plt.title('Episode Lengths over Successive Episodes')
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Length')
    if len(durations)>0:
        plt.ylim(0,max(durations)+1)
    plt.grid()
    plt.plot(range(1,len(durations)+1),durations,'ko',label='Episode Lengths',markersize=5)
    plt.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if timestep%save_step == 0:
        plt.savefig('data/%s_%f_%d_%d_%d.jpg'%(model_name,learning_rate,MAXSTEPS,LAYER_1,LAYER_2))
        plt.show()



if __name__ == '__main__':
    env = OpenAIGym('Acrobot-v1', visualize=False)
    LAYER_1 = 128
    LAYER_2 = 64
    observation = env.reset()
    VPG_agent = VPGAgent(
        states=dict(type='float', shape=env.states['shape']),
        actions=dict(type='int', num_actions=env.actions['num_actions']),
        # discrete action space but continuous state space
        network=[
            dict(type='dense', size=LAYER_1, activation='relu'),  # changed to tanh for best
            dict(type='dense', size=LAYER_2, activation='relu')
        ],
        optimizer=dict(
            type='adam',
            learning_rate=learning_rate

        )
    )

    # Create a Proximal Policy Optimization agent
    PPO_agent = PPOAgent(
        states=dict(type='float', shape=env.states['shape']),
        actions=dict(type='int', num_actions=env.actions['num_actions']),
        # discrete action space but continuous state space
        network=[
            dict(type='dense', size=LAYER_1,activation='relu'), #changed to tanh for best
            dict(type='dense', size=LAYER_2,activation='relu')
        ],
        memory=None, # latest 1000 with default batching
        entropy_regularization=False,
        step_optimizer=dict(
            type='adam',
            learning_rate=learning_rate

        ),

    )


    # for timestep in range(1,20001):
    #     action = PPOagent.act(observation)
    #
    #     observation,done,reward = env.execute(action)
    # if not os.path.exists('models/PPO'):
    #     os.mkdir('models/PPO')
    # if done:
    #     VPGAgent.save_model('models/PPO/%s' % datetime.date)

    #     # Add experience, agent automatically updates model according to batch size
    #     PPOagent.observe(reward=reward, terminal=done)
    #     agg_sum = agg_sum + reward
    #     agg_rewards.append (agg_sum/timestep)
    #     rewards.append(reward)
    #     if timestep % 1000 == 0:
    #         plotter(timestep)
    #
    # agg_rewards = []
    # rewards = []
    # timestep = 0
    startstep = 0
    timestep=1
    model_name = 'VPG'

    while timestep<MAXSTEPS and len(durations)<=100:

        action = VPG_agent.act(observation)
        observation, done, reward = env.execute(action)
        if reward == 0.0:
            ep_done = True
        else:
            ep_done = False
        if ep_done:
            print (timestep-startstep, ' steps needed to finish this episode')
            durations.append(timestep-startstep)
            startstep = timestep + 1
        # Add experience, agent automatically updates model according to batch size
        VPG_agent.observe(reward=reward, terminal=done)
        if ep_done:
            env.reset() #reinitialize
            ep_done = False
        agg_sum = agg_sum + reward
        agg_rewards.append(agg_sum / timestep)
        rewards.append(reward)
        if timestep % 1000 == 0:
            plotter(timestep,model_name=model_name,title='%s Rewards'%model_name,save_step=1000)
        timestep = timestep + 1
    if timestep>MAXSTEPS:
        print ("Timed out. Too many timesteps.")
    else:
        print ("%d episodes finished"%(len(durations)))
    PPO_agent.save_model('models/%s/%f_%d_%d_%d'%(model_name,learning_rate,MAXSTEPS,LAYER_1,LAYER_2))

    print ("Saved model!")
    durations.sort()
    print ("Longest 5 Episode Lengths:")
    print (durations[-5:])
    print ("Shortest 5 Episode Lengths:")
    print (durations[:5])
    print ("Exiting program")
