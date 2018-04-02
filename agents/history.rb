require_relative '../utils/imresize'

# History Class
class History
  def initialize(history_size = 4)
    @history = NMatrix.zeros [history_size, 84, 84], dtype: :float32
  end

  def add(screen)
    @history[0..2, 0..83, 0..83] = @history[1..3, 0..83, 0..83]
    @history[-1, 0..83, 0..83] = screen
  end

  def reset
    @history *= 0
  end

  def get
    @history
  end
end
