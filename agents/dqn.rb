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

    # initial main to target graph
    # nodes = %w[
    #   pred_to_target/l1_w
    #   pred_to_target/l1_b
    #   pred_to_target/l2_w
    #   pred_to_target/l2_b
    #   pred_to_target/l3_w
    #   pred_to_target/l3_b
    #   pred_to_target/l4_w
    #   pred_to_target/l4_b
    #   pred_to_target/q_w
    #   pred_to_target/q_b
    # ]

    # placeholder = @graph.operation(node).output(0)
    # @graph.AddOperation(Tensorflow::OpSpec.new('lAssign', 'Assign', nil, []))

    # nodes.foreach do |node|
    #   hash = {}
    #   placeholder = @graph.operation(node).output(0)
    #   tensor = Tensorflow::Tensor.new(, :float)
    #   @graph.AddOperation(Tensorflow::OpSpec.new('Assign', 'Assign', nil [placeholder], hash))

    #   # q_t, loss = @sess.run(hash,
    # end
  end

  def act(step)
    if rand(0) < @e || step < @config[:pre_train_steps]
      a = rand(@minimal_actions.length)
    else
      hash = {}
      tensor_s_t = Tensorflow::Tensor.new(@history.get.to_a, :float)
      hash[@graph.operation('main/input_data').output(0)] = tensor_s_t
      a = @sess.run(hash, [@graph.operation('main/ArgMax').output(0)], []);
      a = a.flatten[0]
    end
    reward = @env.act(a)
    # obs = @env.get_screen_RGB()
    obs = imresize(@env.get_screen_grayscale(), 84, 84)
    done = @env.game_over()
    [a, obs, reward, done]
  end

  def add_history(obs)
    @history.add(obs)
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
        hash[@graph.operation('target/input_data').output(0)] = tensor_s_t_plus_one
        q_t_plus_one = @sess.run(hash, [@graph.operation('target/add').output(0)], [])
        q_t_plus_one = N[q_t_plus_one].reshape(32, 4)
        terminal = terminal.map { |t| t ? 1 : 0 }
        target_q_t = ((-N[terminal] + 1.0) * 0.99).reshape([32]) * q_t_plus_one.max(1).reshape(32) + NMatrix.new([32], reward)
        hash = {}
        tensor_action = Tensorflow::Tensor.new(action.to_a, :int32)
        tensor_learning_rate_step = Tensorflow::Tensor.new(step, :int64)
        tensor_target_q_t = Tensorflow::Tensor.new(target_q_t.to_a, :float)
        hash[@graph.operation('optimizer/target_q_t').output(0)] = tensor_target_q_t
        hash[@graph.operation('optimizer/action').output(0)] = tensor_action
        hash[@graph.operation('main/input_data').output(0)] = tensor_s_t
        hash[@graph.operation('optimizer/learning_rate_step').output(0)] = tensor_learning_rate_step
        q_t, loss = @sess.run(hash, [@graph.operation('main/add').output(0), @graph.operation('optimizer/loss').output(0)], [])

        @total_loss += loss.first
        @total_q += N[q_t].reshape(32, 4).mean.mean(1).to_f
        @update_count += 1
      end
      update_target_q_network if step % 500 == 499
    end

    # total_loss, total_q, update_count, s1, loss, e = 100, 200, 299, nil, 100, 0.87
    [@total_loss, @total_q, @update_count, obs, loss, @e]
  end

  # TBD
  def update_target_q_network
    nodes = %w[
      pred_to_target/l1_w
      pred_to_target/l1_b
      pred_to_target/l2_w
      pred_to_target/l2_b
      pred_to_target/l3_w
      pred_to_target/l3_b
      pred_to_target/l4_w
      pred_to_target/l4_b
      pred_to_target/q_w
      pred_to_target/q_b
    ]

    l1_w = @sess.run(hash, [@graph.operation('pre/target_l1/w').output(0)], [])
    # tensor = Tensorflow::Tensor.new(l1_w, :float)

    # nodes.each do |node|
    #   hash = {}
    #   placeholder = @graph.operation(node).output(0)
    #   q_t, loss = @sess.run(hash,
    # end

  end

  def play
    a = 0
    hash = {}
    tensor_s_t = Tensorflow::Tensor.new(@history.get.to_a, :float)
    hash[@graph.operation('main/input_data').output(0)] = tensor_s_t
    a = @sess.run(hash, [@graph.operation('main/ArgMax').output(0)], []);
    a = a.flatten[0]
    reward = @env.act(a)
    obs = imresize(@env.get_screen_grayscale(), 84, 84)
    done = @env.game_over()
    [a, obs, reward, done]
  end
end