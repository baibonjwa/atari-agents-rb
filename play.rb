require 'ale_ruby_interface'
require 'pathname'
require 'logger'
require 'rmagick'
require 'pry'
require 'tensorflow'
require 'NMatrix'
require 'ruby-progressbar'

require './utils/recorder'
require './agents/dqn'

TIME_STAMP = Time.now.to_i
RESULT_PATH = "./results/#{TIME_STAMP}".freeze
# LOG_FILE_NAME= "./results/#{TIME_STAMP}/results.log"
# LOG_FILE = File.open(LOG_FILE_NAME, "a")
LOGGER = Logger.new(STDOUT)
RECORD_SPEED = 3
RECORD_WIDTH = 160
RECORD_HEIGHT = 210
Dir.mkdir(RESULT_PATH) unless Dir.exist?(RESULT_PATH)

# Trainning parameter
BATCH_SIZE = 32
UPDATE_FREQ = 4
Y = 0.99
START_E = 1.0
END_E = 0.1
TOTAL_STEPS = 5_000_000
ANNEALING_STEPS = 50_000
NUM_EPISODES = 10_000
# PRE_TRAIN_STEPS = 20_000
PRE_TRAIN_STEPS = 25
MAX_EP_LENGTH = 1000
H_SIZE = 512
TAU = 0.001
INTERVAL = 5
PROGRESS_BAR = ProgressBar.create(total: TOTAL_STEPS, format: '%a %c/%C %e %B %p%% %rit/s %t')

ALE = ALEInterface.new
ALE.set_int('random_seed', 123)
ALE.load_ROM('./atari_roms/breakout.bin')

LEGAL_ACTIONS = ALE.get_legal_action_set()
MINIMAL_ACTIONS = ALE.get_minimal_action_set()

LOGGER.info("Legal Actions: #{LEGAL_ACTIONS}")
LOGGER.info("Minimal Actions: #{MINIMAL_ACTIONS}")

graph = Tensorflow::Graph.new
graph.read_file('dqn-model.pb')
sess_op = Tensorflow::Session_options.new
sess = Tensorflow::Session.new(graph, sess_op)

agent = DQNAgent.new({ batch_size: 32,
                       update_freq: 4,
                       y: 0.99,
                       start_e: 1.0,
                       end_e: 0.1,
                       total_steps: TOTAL_STEPS,
                       annealing_steps: ANNEALING_STEPS,
                       num_episodes: NUM_EPISODES,
                       pre_train_steps: PRE_TRAIN_STEPS,
                       max_ep_length: MAX_EP_LENGTH,
                       h_size: H_SIZE,
                       tau: TAU }, ALE, sess, graph)

# initialize variables
total_reward, reward, avg_reward, max_reward = 0, 0, 0.0
ep_num, ep_rewards, ep_reward, max_ep_reward = 0, [], 0.0

TOTAL_STEPS.times do |step|
  PROGRESS_BAR.increment
  action, obs, reward, done = agent.play
  agent.add_history(obs)
  Recorder.save_screen_png(ALE, step, INTERVAL)
  if done
    ALE.reset_game
    ep_num += 1
    ep_rewards << ep_reward
    ep_reward = 0.0
    Recorder.save_screen_record(RESULT_PATH)
  else
    ep_reward += reward
  end
  total_reward += reward

  next if step < PRE_TRAIN_STEPS
  next if step % 2500 != 2500 - 1

  avg_reward = total_reward / 2500
  avg_loss = total_loss / update_count
  avg_q = total_q / update_count
  max_ep_reward = ep_rewards.max
  min_ep_reward = ep_rewards.min
  avg_ep_reward = ep_rewards.sum.fdiv(ep_rewards.size)

  puts "\navg_r: #{avg_reward},
    avg_l: #{avg_loss}, avg_q: #{avg_q},
    avg_ep_r: #{avg_ep_reward}, max_ep_r: #{max_ep_reward},
    min_ep_r: #{min_ep_reward}, game: #{ep_num}, e: #{e}"

  ep_num = 0
  total_reward = 0
  ep_reward, ep_rewards = 0, []
end