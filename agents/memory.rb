require 'nmatrix'

class Memory
  def initialize(memory_size=10000)
    @memory_size = memory_size
    @memory = []
  end

  def add(screen, reward, action, done)
    @memory.append(NMatrix.new([screen, reward, action, done]))
    if @memory.length > @memory_size
      @memory = self.memory[-@memory_size..0]
    end
  end

  def getState(index, size = 4)
  end

  def sample(size)
  end

  def last(size = 4)
  end

  def count
    return @memory.length
  end
end