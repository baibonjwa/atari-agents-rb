class History
  def initialize
    @history = NMatrix.zeros [4, 84, 84], dtype: :float32
  end

  def add(screen)
    # @history[-1, 0..83, 0..83] = screen
    # @history[-1, 0..83, 0..83] = screen
  end

  def reset
    @history *= 0
  end

  def get
    return @history
  end
end
