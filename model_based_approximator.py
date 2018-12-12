# Imports
import tensorflow as tf
from tensorflow.keras import layers
import gym
import os
import datetime
import numpy as np


env = gym.make('Acrobot-v1')
obs = env.reset()

nn = tf.keras.Sequential()
nn.add(layers.Dense(256, activation='relu',batch_input_shape=(None,6)))
nn.add(layers.Dense(512, activation='relu'))
nn.add(layers.Dense(6, activation=None))


nn.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['mae'])

if __name__=='__main__':
    savedir = 'models/supervised_state_predictor'
    if not os.path.exists('models/supervised_state_predictor'):
        os.mkdir('models/supervised_state_predictor')
    nn.save(filepath=os.path.join(savedir,'superv_1'), overwrite=True)
    state_transition_cache = []
    startStep = 0
    for i in range(100000): #store 100000 state transitions
        action = np.random.randint(low=-1,high=2)
        next_obs,r,d,_ = env.step(action)
        if (d): # done is now True, reset environment and add step
            print (i-startStep, ' steps needed to finish')
            startStep = i + 1
            env.reset()
        state_transition_cache.append([obs, next_obs])
        obs = next_obs

    state_transition_cache = (np.squeeze(np.array(state_transition_cache)))
    np.random.shuffle(state_transition_cache)
    print (state_transition_cache[:,0,:])
    nn.fit(state_transition_cache[:-100,0,:], state_transition_cache[:-100,1,:], epochs=20)
    out = nn.predict(state_transition_cache[-100:,0,:])
    mse = (np.square(state_transition_cache[-100:,1,:]- out)).mean()
    print ('Training mse: ',mse)
    nn.save(filepath=os.path.join(savedir,'superv_2'), overwrite=True)
