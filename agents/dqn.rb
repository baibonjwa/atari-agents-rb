require 'nmatrix'
require_relative 'history'
require_relative 'memory'

# Class DQNAgent
class DQNAgent
  attr_accessor :e

  def initialize(config, env, sess, graph)
    @config = config
    @env = env
    @graph = graph
    @sess = sess
    @legal_actions = env.get_legal_action_set()
    @minimal_actions = env.get_minimal_action_set()
    @history = History.new
    @memory = ReplayMemory.new
    @step_drop = (@config[:start_e] - @config[:end_e]) / @config[:annealing_steps]
    @e = @config[:start_e]
    @total_loss = 0.0
    @total_q = 0.0
    @update_count = 0
  end

  def act(step)
    if rand(0) < @e || step < @config[:pre_train_steps]
      a = rand(@minimal_actions.length)
    else
      hash = {}
      tensor_s_t = Tensorflow::Tensor.new(@history.get.to_a, :float)
      hash[@graph.operation('prediction/s_t').output(0)] = tensor_s_t
      a = @sess.run(hash, [@graph.operation('prediction/ArgMax').output(0)], []);
      a = a.flatten[0]
    end
    reward = @env.act(a)
    # obs = @env.get_screen_RGB()
    obs = imresize(@env.get_screen_grayscale(), 84, 84)
    done = @env.game_over()
    [a, obs, reward, done]
  end

  def learn(step, obs, reward, action, done)
    @history.add(obs)
    @memory.add(obs, reward, action, done)
    loss = 0.0
    if step > @config[:pre_train_steps]
      return if @memory.count < 4
      self.e -= @step_drop if @e > @config[:end_e]
      if (step % @config[:update_freq]).zero?
        s_t, action, reward, s_t_plus_one, terminal = @memory.sample
        hash = {}
        tensor_s_t = Tensorflow::Tensor.new(s_t.to_a, :float)
        tensor_s_t_plus_one = Tensorflow::Tensor.new(s_t_plus_one.to_a, :float)
        hash[@graph.operation('target/target_s_t').output(0)] = tensor_s_t_plus_one
        q_t_plus_one = @sess.run(hash, [@graph.operation('target/target_q/BiasAdd').output(0)], [])
        q_t_plus_one = N[q_t_plus_one].reshape(32, 4)
        terminal = terminal.map { |t| t ? 1 : 0 }
        target_q_t = ((-N[terminal] + 1.0) * 0.99).reshape([32]) * q_t_plus_one.max(1).reshape(32) + NMatrix.new([32], reward)
        hash = {}
        tensor_action = Tensorflow::Tensor.new(action.to_a, :int64)
        tensor_learning_rate_step = Tensorflow::Tensor.new(step, :int64)
        tensor_target_q_t = Tensorflow::Tensor.new(target_q_t.to_a, :float)
        hash[@graph.operation('optimizer/target_q_t').output(0)] = tensor_target_q_t
        hash[@graph.operation('optimizer/action').output(0)] = tensor_action
        hash[@graph.operation('prediction/s_t').output(0)] = tensor_s_t
        hash[@graph.operation('optimizer/learning_rate_step').output(0)] = tensor_learning_rate_step
        q_t, loss = @sess.run(hash, [@graph.operation('prediction/q/BiasAdd').output(0), @graph.operation('optimizer/loss').output(0)], [])

        @total_loss += loss.first
        @total_q += N[q_t].reshape(32, 4).mean.mean(1).to_f
        @update_count += 1
      end
      update_target_q_network if step % 500 == 499
    end

    # total_loss, total_q, update_count, s1, loss, e = 100, 200, 299, nil, 100, 0.87
    [@total_loss, @total_q, @update_count, obs, loss, @e]
  end

  def update_target_q_network
    nodes = %w[
      prediction/l1
      prediction/l1_b
      prediction/l2_w
      prediction/l2_b
      prediction/l3_w
      prediction/l3_b
      prediction/l4_w
      prediction/l4_b
      prediction/q_w
      prediction/q_b
      target/target_l1
      target/target_l1_b
      target/target_l2_w
      target/target_l2_b
      target/target_l3_w
      target/target_l3_b
      target/target_l4_w
      target/target_l4_b
      target/target_q_w
      target/target_q_b
    ]
  end

  def play
  end

end