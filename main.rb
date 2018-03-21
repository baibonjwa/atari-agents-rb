require 'ale_ruby_interface'
require 'pathname'
require 'logger'
require 'rmagick'
require 'pry'

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

ALE = ALEInterface.new
ALE.set_int('random_seed', 123)
ALE.load_ROM('./atari_roms/breakout.bin')

LEGAL_ACTIONS = ALE.get_legal_action_set()
MINIMAL_ACTIONS = ALE.get_minimal_action_set()

LOGGER.info("Legal Actions: #{LEGAL_ACTIONS}")
LOGGER.info("Minimal Actions: #{MINIMAL_ACTIONS}")

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
