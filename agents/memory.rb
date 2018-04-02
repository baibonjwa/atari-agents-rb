require 'nmatrix'

# Class ReplayMemory
class ReplayMemory

  attr_accessor :count

  def initialize(memory_size = 10_000)
    @memory_size = memory_size
    @actions = NMatrix.zeros [memory_size], dtype: :int64
    @rewards = NMatrix.zeros [memory_size], dtype: :int64
    @screens = NMatrix.zeros [memory_size, 84, 84], dtype: :float32
    @terminals = NMatrix.new [memory_size], dtype: :object
    @history_length = 4
    @batch_size = 32
    @count = 0
    @current = 0
    @memory = []
    @prestates = NMatrix.zeros [@batch_size, 4, 84, 84], dtype: :float32
    @poststates = NMatrix.zeros [@batch_size, 4, 84, 84], dtype: :float32
  end

  def add(screen, reward, action, done)
    @actions[@current] = action
    @rewards[@current] = reward
    @screens[@current, 0..83, 0..83] = screen
    @terminals[@current] = done
    @count = [@count, @current + 1].max
    @current = (@current + 1) % @memory_size
  end

  def getState(index)
    index = index % @count
    @screens[(index - (@history_length - 1))..(index + 1), 0..83, 0..83]
  end

  def sample
    indexes = []
    while indexes.length < @batch_size
      while true
        index = rand(@history_length..@count - 1)
        next if index >= @current && index - @history_length < @current
        next if @terminals[(index - @history_length)..index].any?
        break
      end
      @prestates[indexes.length, 0..3, 0..83, 0..83] = getState(index - 1)
      @poststates[indexes.length, 0..3, 0..83, 0..83] = getState(index)
      indexes << index
    end
    actions = @actions.to_a.values_at(*indexes)
    rewards = @rewards.to_a.values_at(*indexes)
    terminals = @terminals.to_a.values_at(*indexes)
    [@prestates.transpose([0, 2, 3, 1]), actions, rewards, @poststates.transpose([0, 2, 3, 1]), terminals]
  end
end
