
# INITIALIZATION: libraries, parameters, network...
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves 
import numpy as np
from keras.layers import Dense,Dropout
from tqdm import tqdm
from keras import optimizers
np.random.seed(7)
import random
from collections import Counter


import gym
env = gym.make('CartPole-v0')


model = Sequential()
model.add(Dense(50, input_dim=4,activation='relu'))
model.add(Dense(256,kernel_initializer='normal', activation='relu'))
Dropout(0.8)
model.add(Dense(128,kernel_initializer='normal', activation='relu'))
Dropout(0.8)
model.add(Dense(512,kernel_initializer='normal', activation='relu'))
Dropout(0.8)
model.add(Dense(128,kernel_initializer='normal', activation='relu'))
Dropout(0.8)
model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))    # Same number of outputs as possible actions




observetime = 500                        # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7                            # Probability of doing a random move
score_requirement = 50                   # score requirement  




# create training datasets correspoding to states and actions

def population(initial_games): 
    accepted_scores=[]
    training_data=[]
    scores=[]
    for _ in tqdm(range(initial_games)):
        observation = env.reset()
        obs = observation
        state = obs
        score=0
        game_memory=[]
        prev_observation=[]
        for _ in range(observetime):           
            env.render()
            if np.random.rand() <= epsilon:
                action = random.randint(0,1)
            else:
                state = np.squeeze(state).reshape(1,4)
                Q = model.predict(state)        
                action = np.int(np.round(Q))
            observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
            obs_new = observation_new          # (Formatting issues)
            state_new = obs_new     # Update the input with the new state of the game
            if len(prev_observation)>0:
                game_memory.append([prev_observation,action])
            state = state_new         # Update state
            prev_observation = state_new
            score+=1
            if done:
                break
    
        if score>=score_requirement:
            accepted_scores.append(scores)
            for data in game_memory:
                if data[1]==1:
                    output=[1]
                else: 
                    output=[0]
                
                training_data.append([data[0],output])
                                
                
        env.reset()
        scores.append(score)
    
    np.save('mem.npz',training_data)
    print("Avg accepted score: ",np.mean(accepted_scores))
    print("Median accepted score: ",np.median(accepted_scores))
    env.close()     
    
    return training_data




training_data = population(10000)



X = np.array([i[0] for i in training_data])
y = [i[1] for i in training_data]


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,y,batch_size=100,epochs=30)


def play():#import gym                                # To train our network
    env = gym.make('CartPole-v0') 
    observation = env.reset()
    obs = observation
    state = obs
    done = False
    tot_reward = 0.0
    while not done:
        env.render()                    # Uncomment to see game running
        state = np.squeeze(state).reshape(1,4)
        Q = model.predict(state)        
        action = np.int(np.round(Q))
        observation, reward, done, info = env.step(action)
        obs = observation
        state = obs    
        tot_reward += reward
    env.close()
    print('Game ended! Total reward: {}'.format(tot_reward))
    return tot_reward

play()





