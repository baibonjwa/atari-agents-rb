require_relative '../utils/imresize'

# History Class
class History
  def initialize(history_size = 4)
    @history = NMatrix.zeros [84, 84, history_size], dtype: :float32
  end

  def add(screen)
    @history[0..83, 0..83, 0..2] = @history[0..83, 0..83, 1..3]
    @history[0..83, 0..83, -1] = screen
  end

  def reset
    @history *= 0
  end

  def get
    @history.reshape([1, 84, 84, 4])
  end
end
