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
  end

  def act(step)
    # binding.pry
    if rand(0) < @e || step < @config[:pre_train_step]
      a = rand(@legal_actions.length)
    else
      hash = {}
      hash[@graph.operation('prediction/s_t')] = @history.get
      a = @sess.run(hash, [graph.operation('prediction/ArgMax').output(0)], []);
    end
    # a = rand(@legal_actions.length)
    reward = @env.act(a)
    # obs = @env.get_screen_RGB()
    obs = @env.get_screen_grayscale()

    done = @env.game_over()
    [a, obs, reward, done]
  end

  def learn(step, obs, reward, action, done)
    @history.add(obs)
    @memory.add(obs, reward, action, done)
    loss = 0.0
    if step > @config[:pre_train_step]
      return if @memory.count < 4
      if @e > @config[:endE]
        self.e -= @step_drop
      elsif step % @config[:update_freq].zero?
        s_t, action, reward, s_t_plus_1, terminal = @memory.sample()
        hash = {}
        hash[@graph.operation('target/target_s_t')] = s_t_plus_1
        q_t_plus_1 = @sess.run(hash, [graph.operation('target/target_q/BiasAdd')])
        target_q_t = (1.0 - terminal) * 0.99 * q_t_plus_1.max + reward
        hash = {}
        hash[@graph.operation('optimizer/target_q_t')] = target_q_t
        hash[@graph.operation('optimizer/action')] = action
        hash[@graph.operation('prediction/s_t')] = s_t
        hash[@graph.operation('optimizer/learning_rate_step')] = step
        q_t, loss = @sess.run(
          hash,
          [
            graph.operation('prediction/q/BiasAdd'),
            graph.operation('optimizer/loss')
          ]
        )
        @total_loss += loss
        @total_q += q_t.mean
        @update_count += 1
      end
      update_target_q_network if step % 500 == 499
    end

    # total_loss, total_q, update_count, s1, loss, e = 100, 200, 299, nil, 100, 0.87
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