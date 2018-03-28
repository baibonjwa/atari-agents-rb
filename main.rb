require 'ale_ruby_interface'
require 'pathname'
require 'logger'
require 'rmagick'
require 'pry'
require 'tensorflow'
require './agents/dqn'

include Magick

TIME_STAMP = Time.now.to_i
RESULT_PATH = "./results/#{TIME_STAMP}"
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
TOTAL_STEPS = 5000000
ANNEALING_STEPS = 50000
NUM_EPISODES = 10000
PRE_TRAIN_STEPS = 20000
MAX_EP_LENGTH = 1000
H_SIZE = 512
TAU = 0.001

ALE = ALEInterface.new
ALE.set_int('random_seed', 123)
ALE.load_ROM('./atari_roms/breakout.bin')

LEGAL_ACTIONS = ALE.get_legal_action_set()
MINIMAL_ACTIONS = ALE.get_minimal_action_set()

LOGGER.info("Legal Actions: #{LEGAL_ACTIONS}")
LOGGER.info("Minimal Actions: #{MINIMAL_ACTIONS}")


scope = Tensorflow::Scope.new('steps')
graph = scope.graph
graph.read_file("dqn.pb")
session_op = Tensorflow::Session_options.new
session = Tensorflow::Session.new(graph, session_op)
# hash = {}
# hash[step_input] = tensor_1
# hash[placeholder_1] = tensor_1
# hash[placeholder_2] = tensor_2
# out_tensor = session.run(hash, [op.output(0)], [])
# out_tensor = session.run(hash, [output], [])
# binding.pry
# puts out_tensor[0]
# graph.write_file("results.pb")
# system "python tensorboard.py `pwd`/logs"
# out_tensor = session.run({}, [output], [])

# agent = Agent.new({
#   batch_size: 32,
#   update_freq: 4,
#   y: 0.99,
#   start_e: 1.0,
#   end_e: 0.1,
#   total_steps: 5000000,
#   annealing_steps: 50000,
#   num_episodes: 10000,
#   pre_train_steps: 20000,
#   max_ep_length: 1000,
#   h_size: 512,
#   tau: 0.001,
# }, ALE, scope)
# agent.train()

total_reward = 0
frame = 0
while !ALE.game_over()
  action = LEGAL_ACTIONS[Random.rand(LEGAL_ACTIONS.length)]
  reward = ALE.act(action)
  frame = frame + 1
  if frame % RECORD_SPEED == 0
    ALE.save_screen_PNG("./results/#{TIME_STAMP}/#{(Time.now.to_f * 10000).to_i}.png")
  end
  total_reward += reward
end

images = Dir["./results/#{TIME_STAMP}/*"]
images.each do |image|
  i = Magick::Image.read(image).first
  i = i.resize(RECORD_WIDTH, RECORD_HEIGHT)
  i.write(Pathname(image).sub_ext('.jpg')) do
    self.format='JPEG'
    self.quality=80
  end
end

sequence = ImageList.new(*Dir["#{RESULT_PATH}/*.jpg"].sort)
sequence.delay = 2
sequence.ticks_per_second = 60
sequence.write("#{RESULT_PATH}/results.mp4")
sequence.write("#{RESULT_PATH}/results.gif")

FileUtils.rm_f(Dir["#{RESULT_PATH}/*.jpg"])
FileUtils.rm_f(Dir["#{RESULT_PATH}/*.png"])

LOGGER.info("Episode ended with score: #{total_reward}")
