require 'nmatrix'
require_relative 'history'
require_relative 'memory'

class DQNAgent
  attr_accessor :e

  def initialize(config, env, sess, graph)
    @config = config
    @env = env
    @graph = graph
    @legal_actions = env.get_legal_action_set()
    @minimal_actions = env.get_minimal_action_set()
    @history = History.new
    @memory = ReplayMemory.new
  end

  def act
    # if rand(0) < @e || step < @config[:pre_train_step]
    #   a = rand(@legal_actions.length)
    # else
    #   hash = {}
    #   a = @sess.run(hash, [graph.operation('prediction/ArgMax').output(0)], []);
    # end
    a = rand(@legal_actions.length)
    reward = @env.act(a)
    obs = @env.get_screen_RGB()
    done = @env.game_over()
    [a, obs, reward, done]
  end

  def learn(step, obs, reward, action, done)
    total_loss, total_q, update_count, s1, loss, e = 100, 200, 299, nil, 100, 0.87
    [total_loss, total_q, update_count, s1, loss, e]
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