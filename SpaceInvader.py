import logging
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import gym
#import Image

# Import the renderer as it's what we'll be using to plot 
from gym.envs.classic_control import rendering 

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    env = gym.make('SpaceInvaders-v0' if len(sys.argv)<2 else sys.argv[1])

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    #env.monitor.start(outdir, force=True, seed=0)
    #gym.wrappers.Monitor(env, outdir)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    agent = RandomAgent(env.action_space)

    episode_count = 200
    max_steps = 200
    reward = 0
    done = False

    # grab an instance of the image viewer
    viewer = rendering.SimpleImageViewer()

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

            # the upscaling happens here 

            # grab the rgb array
            rgb = env.render('rgb_array')
            # upscale
            upscaled = repeat_upsample(rgb, 4, 4)
            # plot using the default viewer
            viewer.imshow(upscaled)


    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")

gym.upload(outdir)
