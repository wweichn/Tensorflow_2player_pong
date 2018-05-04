#!/usr/bin/python
#-*- coding:utf-8 -*-
from xitari_python_interface import ALEInterface, ale_fillRgbFromPalette
import pygame
import numpy as np
from numpy.ctypeslib import as_ctypes
from dqn.agent_dqn import *
from dqn.game_screen import GameScreen
from dqn.scale import scale_image
from dqn import config
from tqdm import tqdm
import tensorflow as tf

# Converts the palette values to RGB values

Game_action1 = [0,1,3,4]
Game_action2 = [20,21,23,24]


def getRgbFromPalette(ale, surface, rgb_new):
    # Environment parameters
    width = ale.ale_getScreenWidth()
    height = ale.ale_getScreenHeight()

    # Get current observations
    obs = np.zeros(width * height, dtype=np.uint8)
    n_obs = obs.shape[0]
    ale.ale_fillObs(as_ctypes(obs), width * height)

    # Get RGB values of values
    n_rgb = n_obs * 3
    rgb = np.zeros(n_rgb, dtype=np.uint8)
    ale_fillRgbFromPalette(as_ctypes(rgb), as_ctypes(obs), n_rgb, n_obs)

    # Convert uint8 array into uint32 array for pygame visualization
    for i in range(n_obs):
        # Convert current pixel into RGBA format in pygame
        cur_color = pygame.Color(int(rgb[i]), int(rgb[i + n_obs]), int(rgb[i + 2 * n_obs]))
        cur_mapped_int = surface.map_rgb(cur_color)
        rgb_new[i] = cur_mapped_int

    # Reshape and roll axis until it fits imshow dimensions
    return np.rollaxis(rgb.reshape(3, height, width), axis=0, start=3)

def wf(str, flag):
    if flag == 0:
        filepath = './logs/file' + '/paddle_bounce.txt'
    elif flag == 1:
        filepath = './logs/file' + '/wall_bounce.txt'
    else:
        filepath = './logs/file' + '/serving_time'
    F_epoch = open(filepath, 'a')
    F_epoch.write(str + '\n')
    F_epoch.close()


'''
if(len(sys.argv) < 2):
    print("Usage ./random_agent.py <ROM_FILE_NAME>")
    sys.exit()
'''
def main():

    sess = tf.InteractiveSession()

    roms = 'roms/Pong2Player025.bin'
    ale = ALEInterface(roms.encode('utf-8'))
    width = ale.ale_getScreenWidth()
    height = ale.ale_getScreenHeight()
    conf = config.Config()
    # Reset game

    ale.ale_resetGame()
    (display_width, display_height) = (width * 2, height * 2)

    # Initialize Pygame
    pygame.init()
    screen_ale = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Arcade Learning Environment Random Agent Display")
    pygame.display.flip()

    game_surface = pygame.Surface((width, height), depth=8)

    # Initialize GameScreen object for framepooling
    game_screen = GameScreen()

    # init clock
    clock = pygame.time.Clock()

    # Clear screen
    screen_ale.fill((0, 0, 0))

    agent1 = Agent_dqn(sess,conf,"agent1")
    agent2 = Agent_dqn(sess,conf,"agent2")

    sess.run(tf.initialize_all_variables())
    agent1.load_model()
    history = History(conf)
    ep_rewards1 = []
    ep_rewards2 = []
    #poch = 0

    for epoch in range(conf.max_epoch):
        start_step = 0

        num_game, ep_reward1, ep_reward2, total_loss = 0, 0, 0, 0
        total_reward1, total_reward2, agent1.total_loss, agent2.total_loss = 0,0,0,0
        max_avg_ep_reward1, max_avg_ep_reward2 = 0,0
        ep_rewards1, ep_rewards2 = [], []



        # Both agents perform random actions
        # Agent A : [NOOP, FIRE, RIGHT, LEFT]
        # Agent B : [NOOP, FIRE, RIGHT, LEFT]
        # ale.ale_act2(np.random.choice([0, 1, 3, 4]), np.random.choice([20, 21, 23, 24]))

        # Fill buffer of game screen with current frame
        numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
        rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
        del numpy_surface

        game_screen.paint(rgb)
        pooled_screen = game_screen.grab()
        scaled_pooled_screen = scale_image(pooled_screen)

        for _ in range(conf.history_length):
            history.add(scaled_pooled_screen)

        #  初始时length = 4的screen

        for agent1.step in tqdm(range(conf.max_step), ncols = 70, initial = start_step):
            #显示进度
            agent2.step = agent1.step
            #print("agent1.step", agent1.step)

            if agent1.step == conf.learn_start:
                num_game, ep_reward1, ep_reward2, total_loss = 0, 0, 0, 0
                total_reward1, total_reward2, agent1.total_loss, agent2.total_loss = 0,0,0,0
                max_avg_ep_reward1, max_avg_ep_reward = 0,0
                ep_rewards1, ep_rewards2 = [], []



            # predict
            action1_index = agent1.select_action(history.get())
            action2_index = agent2.select_action(history.get())

            # act
            #print(action1_index)
            #print(action2_index)
            action1 = Game_action1[action1_index]
            action2 = Game_action2[action2_index]

            ale.ale_act2(action1, action2)
            #ale.ale_act2(4,24)
            terminal = ale.ale_isGameOver()
            if agent1.step == conf.max_step - 1:
                terminal = True
            reward1 = ale.ale_getRewardA()
            reward2 = ale.ale_getRewardB()

            # Fill buffer of game screen with current frame
            numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)

            rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
            del numpy_surface
            game_screen.paint(rgb)
            pooled_screen = game_screen.grab()
            scaled_pooled_screen = scale_image(pooled_screen)

            history.add(scaled_pooled_screen)
            agent1.perceive(scaled_pooled_screen,reward1,action1_index, terminal)
            agent2.perceive(scaled_pooled_screen,reward2, action2_index, terminal)


            # Print frame onto display screen
            screen_ale.blit(pygame.transform.scale2x(game_surface), (0, 0))

            if terminal:
                print("terminal")
                ale.ale_resetGame()
                num_game += 1
                ep_rewards1.append(ep_reward1)
                ep_rewards2.append(ep_reward2)
                '''
                print("reward1", np.mean(ep_reward1))
                print("reward2", np.mean(ep_reward2))
                '''
                ep_reward1 = 0
                ep_reward2 = 0
                 # Fill buffer of game screen with current frame
                numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)

                rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
                del numpy_surface
                game_screen.paint(rgb)
                pooled_screen = game_screen.grab()
                scaled_pooled_screen = scale_image(pooled_screen)
                #end of an episode

            else:
                ep_reward1 += reward1
                ep_reward2 += reward2

            # Update the display screen
            pygame.display.flip()
            total_reward1 += reward1
            total_reward2 += reward2

            # do a test to get statistics so far
            if agent1.step >= conf.learn_start:
                if agent1.step % conf.test_step == conf.test_step - 1:
                    avg_reward1 = total_reward1 / conf.test_step
                    avg_reward2 = total_reward2 / conf.test_step
                    avg_loss1 = agent1.total_loss / agent1.update_count
                    avg_loss2 = agent2.total_loss / agent2.update_count
                    avg_q1 = agent1.total_q / agent1.update_count
                    avg_q2 = agent2.total_q / agent2.update_count

                    try:
                        max_ep_reward1 = np.max(ep_rewards1)
                        min_ep_reward1 = np.min(ep_rewards1)
                        avg_ep_reward1 = np.mean(ep_rewards1)
                        max_ep_reward2 = np.max(ep_rewards2)
                        min_ep_reward2 = np.min(ep_rewards2)
                        avg_ep_reward2 = np.mean(ep_rewards2)

                    except:
                        max_ep_reward1, min_ep_reward1, avg_ep_reward1, max_ep_reward2, min_ep_reward2, avg_ep_reward2 = 0,0,0,0,0,0



                    print('\nFor Agent A at Epoch %d: Step %d: avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                  % (epoch, agent1.step, avg_reward1, avg_loss1, avg_q1, avg_ep_reward1, max_ep_reward1, min_ep_reward1, num_game))
                    print('\nFor Agent B at Epoch %d: Step %d: avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                  % (epoch, agent2.step, avg_reward2, avg_loss2, avg_q2, avg_ep_reward2, max_ep_reward2, min_ep_reward2, num_game))

                    agent1.save_model()

                    max_avg_ep_reward1 = max(max_avg_ep_reward1, avg_ep_reward1)

                    agent2.save_model()

                    max_avg_ep_reward2 = max(max_avg_ep_reward2, avg_ep_reward2)

                    agent1.inject_summary({
                    'average.reward': avg_reward1,
                    'average.loss': avg_loss1,
                    'average.q': avg_q1,
                    'episode.max_reward': max_ep_reward1,
                    'episode.min_reward': min_ep_reward1,
                    'episode.avg_reward': avg_ep_reward1,
                    'episode.num_of_game': num_game,
                    'episode.rewards': ep_rewards1,
                   })

                    agent2.inject_summary({
                    'average.reward': avg_reward2,
                    'average.loss': avg_loss2,
                    'average.q': avg_q2,
                    'episode.max_reward': max_ep_reward2,
                    'episode.min_reward': min_ep_reward2,
                    'episode.avg_reward': avg_ep_reward2,
                    'episode.num_of_game': num_game,
                    'episode.rewards': ep_rewards2,
                   })

                    num_game = 0
                    total_reward1, total_reward2 = 0,0
                    agent1.total_loss, agent2.total_loss = 0,0
                    agent1.total_q, agent2.total_q = 0,0
                    agent1.update_count, agent2.update_count = 0,0
                    ep_reward1, ep_reward2 = 0,0
                    ep_rewards1, ep_rewards2 = [],[]

        total_points, paddle_bounce, wall_bounce, serving_time = [], [], [], []
        for _ in range(5):
            cur_total_points, cur_paddle_bounce, cur_wall_bounce, cur_serving_time = 0, 0, 0, 0

            # Restart game
            ale.ale_resetGame()

          # Get first frame of gameplay
            numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
            rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
            del numpy_surface
            game_screen.paint(rgb)
            pooled_screen = game_screen.grab()
            scaled_pooled_screen = scale_image(pooled_screen)

            # Create history for testing purposes
            test_history = History(conf)

            # Fill first 4 images with initial screen
            for _ in range(conf.history_length):
                test_history.add(scaled_pooled_screen)

            while not ale.ale_isGameOver():
            # 1. predict
                action1 = agent1.select_action(test_history.get(),0)
                action2 = agent2.select_action(test_history.get(),0)

            # 2. act
                ale.ale_act2(action1, action2)
                terminal = ale.ale_isGameOver()
                reward1 = ale.ale_getRewardA()
                reward2 = ale.ale_getRewardB()

            # Record game statistics of current episode
                cur_total_points = ale.ale_getPoints()
                cur_paddle_bounce = ale.ale_getSideBouncing()
                if ale.ale_getWallBouncing():
                    cur_wall_bounce += 1
                if ale.ale_getServing():
                    cur_serving_time += 1

            # Fill buffer of game screen with current frame
                numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
                rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
                del numpy_surface
                game_screen.paint(rgb)
                pooled_screen = game_screen.grab()
                scaled_pooled_screen = scale_image(pooled_screen)
              #  agent1.perceive(reward1, action1, terminal)
              #  agent2.perceive(reward2, action2, terminal)

                # Print frame onto display screen
                screen_ale.blit(pygame.transform.scale2x(game_surface), (0, 0))

                # Update the display screen
                pygame.display.flip()

                # Append current episode's statistics into list
        #total_points.append(cur_total_points)
        paddle_bounce = cur_paddle_bounce / cur_total_points
        if cur_paddle_bounce == 0:
            wall_bounce = cur_wall_bounce / (cur_paddle_bounce + 1)
        else:
            wall_bounce = cur_wall_bounce / cur_paddle_bounce
        serving_time = cur_serving_time / cur_total_points
        wf(paddle_bounce,0)
        wf(wall_bounce,1)
        wf(serving_time,2)


        # Save results of test after current epoch
        '''
            cur_paddle_op = agent1.paddle_op.eval()
            cur_paddle_op[agent1.epoch] = sum(paddle_bounce) / len(paddle_bounce)
            agent1.paddle_assign_op.eval({agent1.paddle_input: cur_paddle_op})

            cur_wall_op = agent1.wall_op.eval()
            cur_wall_op[agent1.epoch] = sum(wall_bounce) / len(wall_bounce)
            agent1.wall_assign_op.eval({agent1.wall_input: cur_wall_op})

            cur_serving_op = agent1.serving_op.eval()
            cur_serving_op[agent1.epoch] = sum(serving_time) / len(serving_time)
            agent1.serving_assign_op.eval({agent1.serving_input: cur_serving_op})
        '''
        agent1.save_model_epoch(epoch)
        agent2.save_model_epoch(epoch)

        # delay to 60fps
        #clock.tick(60.)

# Print out result of the episode
# Properly close display
    '''
    pygame.display.quit()
    pygame.quit()
    '''
if __name__ == "__main__":
    main()

