require 'ale_ruby_interface'
require 'pathname'
require 'logger'
require 'rmagick'
require 'pry'
require 'tensorflow'
require './agents/dqn'
require 'NMatrix'

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


graph = Tensorflow::Graph.new
graph.read_file("dqn-frozen.pb")
session_op = Tensorflow::Session_options.new
session = Tensorflow::Session.new(graph, session_op)
hash = {}
action = N[1, 1, 3, 3, 1, 1, 1, 2, 1, 0, 3, 2, 0, 1, 3, 3, 0, 1, 0, 2, 3, 0, 2, 2, 3, 0, 0, 1, 0, 2, 2, 3]
target_q_t = N[
  -0.07291178, -0.08841241, -0.12142662, -0.05167786, -0.05393076,
  -0.08957968, -0.11104095, -0.10681677, -0.04481473, -0.10269744,
  -0.13945529, -0.07953419, -0.04515881, -0.0472517 , -0.09946557,
  -0.096865  , -0.04165004, -0.10311015, -0.06276112, -0.14854823,
  -0.07176998, -0.09394912, -0.05810888, -0.05962679, -0.11295396,
  -0.06504908, -0.03945166, -0.12991059,  0.90407543, -0.08911409,
  0.0, -0.10775804
]
# hash[graph.operation('main/s_t').output(0)] = ALE.get_screen_RGB()
s_t = NMatrix.zeros([32, 84, 84, 4], dtype: :float32)
# tensor_s_t = Tensorflow::Tensor.new(ALE.get_screen_RGB().to_a, :float)
tensor_s_t = Tensorflow::Tensor.new(s_t.to_a, :float)
tensor_action = Tensorflow::Tensor.new(action.to_a, :int64)
tensor_target_q_t = Tensorflow::Tensor.new(target_q_t.to_a, :float)
tensor_learning_rate_step = Tensorflow::Tensor.new(2504, :int64)

hash[graph.operation('prediction/s_t').output(0)] = tensor_s_t
hash[graph.operation('optimizer/action').output(0)] = tensor_action
hash[graph.operation('optimizer/target_q_t').output(0)] = tensor_target_q_t
hash[graph.operation('optimizer/learning_rate_step').output(0)] = tensor_learning_rate_step

# q = session.run(hash, [graph.operation('prediction/q/BiasAdd').output(0)], []);
begin
  session.run(hash, [graph.operation('optimizer/loss').output(0)], []);
rescue Exception => ex
  system "python tensorboard.py `pwd`/logs"
  graph.write_file("results.pb")
  puts ex
end

# inputs.each do |port, tensor|
#   puts port.c
#   inputPorts.push(port.c)
#   inputValues.push(tensor.tensor)
# end

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

# total_reward = 0
# frame = 0
# while !ALE.game_over()
#   action = LEGAL_ACTIONS[Random.rand(LEGAL_ACTIONS.length)]
#   reward = ALE.act(action)
#   frame = frame + 1
#   if frame % RECORD_SPEED == 0
#     ALE.save_screen_PNG("./results/#{TIME_STAMP}/#{(Time.now.to_f * 10000).to_i}.png")
#   end
#   total_reward += reward
# end

# images = Dir["./results/#{TIME_STAMP}/*"]
# images.each do |image|
#   i = Magick::Image.read(image).first
#   i = i.resize(RECORD_WIDTH, RECORD_HEIGHT)
#   i.write(Pathname(image).sub_ext('.jpg')) do
#     self.format='JPEG'
#     self.quality=80
#   end
# end

# sequence = ImageList.new(*Dir["#{RESULT_PATH}/*.jpg"].sort)
# sequence.delay = 2
# sequence.ticks_per_second = 60
# sequence.write("#{RESULT_PATH}/results.mp4")
# sequence.write("#{RESULT_PATH}/results.gif")

# FileUtils.rm_f(Dir["#{RESULT_PATH}/*.jpg"])
# FileUtils.rm_f(Dir["#{RESULT_PATH}/*.png"])

# LOGGER.info("Episode ended with score: #{total_reward}")
