require 'nmatrix'
require_relative 'history'
require_relative 'memory'

class Agent
  def initialize(config, env, scope)
    @config = config
    @env = env
    @sess = scope

    @history = History.new(config)
    @memory = ReplayMemory.new(config)
  end

  def train
  end

  def predict
  end

  def observe
  end

  def q_learning_mini_batch
  end

  def build_dqn
  end

  def update_target_q_network
  end

  def save_weight_to_pkl
  end

  def load_weight_from_pkl
  end

  def inject_summary
  end

  def play
  end

end