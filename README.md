## atari-agent-rb
tensorflow.rb implementation of DQN with ruby(experiment)

This implementation contains:

#### DQN
1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
5. Dueling DQN

#### Model
dqn-init.pb is the model with initial values
dqn-model.pb is the model with cpu trainning 1h

## Requirements

- tensorflow.rb
- ale_ruby_interface
- NMatrix
- ruby-progressbar
- rmagick

## Usage

```shell
  ruby main.rb
```

Play:

```shell
  ruby play.rb
```

## Results

No results because very poor performance.


## License

MIT License.
