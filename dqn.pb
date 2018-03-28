node {
  name: "step/step/initial_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "step/step"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "step/step/Assign"
  op: "Assign"
  input: "step/step"
  input: "step/step/initial_value"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@step/step"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "step/step/read"
  op: "Identity"
  input: "step/step"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@step/step"
      }
    }
  }
}
node {
  name: "step/step_input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "step/Assign"
  op: "Assign"
  input: "step/step"
  input: "step/step_input"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@step/step"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/s_t"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 84
        }
        dim {
          size: 84
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "main/l1/w/Initializer/truncated_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\010\000\000\000\010\000\000\000\004\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "main/l1/w/Initializer/truncated_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l1/w/Initializer/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "main/l1/w/Initializer/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "main/l1/w/Initializer/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 10
    }
  }
}
node {
  name: "main/l1/w/Initializer/truncated_normal/mul"
  op: "Mul"
  input: "main/l1/w/Initializer/truncated_normal/TruncatedNormal"
  input: "main/l1/w/Initializer/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
}
node {
  name: "main/l1/w/Initializer/truncated_normal"
  op: "Add"
  input: "main/l1/w/Initializer/truncated_normal/mul"
  input: "main/l1/w/Initializer/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
}
node {
  name: "main/l1/w"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 8
        }
        dim {
          size: 8
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l1/w/Assign"
  op: "Assign"
  input: "main/l1/w"
  input: "main/l1/w/Initializer/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l1/w/read"
  op: "Identity"
  input: "main/l1/w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
}
node {
  name: "main/l1/Conv2D"
  op: "Conv2D"
  input: "main/s_t"
  input: "main/l1/w/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 4
        i: 4
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "main/l1/biases/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l1/biases"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l1/biases/Assign"
  op: "Assign"
  input: "main/l1/biases"
  input: "main/l1/biases/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l1/biases/read"
  op: "Identity"
  input: "main/l1/biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
}
node {
  name: "main/l1/BiasAdd"
  op: "BiasAdd"
  input: "main/l1/Conv2D"
  input: "main/l1/biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "main/Relu"
  op: "Relu"
  input: "main/l1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/l2/w/Initializer/truncated_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "main/l2/w/Initializer/truncated_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l2/w/Initializer/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "main/l2/w/Initializer/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "main/l2/w/Initializer/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 26
    }
  }
}
node {
  name: "main/l2/w/Initializer/truncated_normal/mul"
  op: "Mul"
  input: "main/l2/w/Initializer/truncated_normal/TruncatedNormal"
  input: "main/l2/w/Initializer/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
}
node {
  name: "main/l2/w/Initializer/truncated_normal"
  op: "Add"
  input: "main/l2/w/Initializer/truncated_normal/mul"
  input: "main/l2/w/Initializer/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
}
node {
  name: "main/l2/w"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l2/w/Assign"
  op: "Assign"
  input: "main/l2/w"
  input: "main/l2/w/Initializer/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l2/w/read"
  op: "Identity"
  input: "main/l2/w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
}
node {
  name: "main/l2/Conv2D"
  op: "Conv2D"
  input: "main/Relu"
  input: "main/l2/w/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "main/l2/biases/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l2/biases"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l2/biases/Assign"
  op: "Assign"
  input: "main/l2/biases"
  input: "main/l2/biases/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l2/biases/read"
  op: "Identity"
  input: "main/l2/biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
}
node {
  name: "main/l2/BiasAdd"
  op: "BiasAdd"
  input: "main/l2/Conv2D"
  input: "main/l2/biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "main/Relu_1"
  op: "Relu"
  input: "main/l2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/l3/w/Initializer/truncated_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "main/l3/w/Initializer/truncated_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l3/w/Initializer/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "main/l3/w/Initializer/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "main/l3/w/Initializer/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 42
    }
  }
}
node {
  name: "main/l3/w/Initializer/truncated_normal/mul"
  op: "Mul"
  input: "main/l3/w/Initializer/truncated_normal/TruncatedNormal"
  input: "main/l3/w/Initializer/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
}
node {
  name: "main/l3/w/Initializer/truncated_normal"
  op: "Add"
  input: "main/l3/w/Initializer/truncated_normal/mul"
  input: "main/l3/w/Initializer/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
}
node {
  name: "main/l3/w"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l3/w/Assign"
  op: "Assign"
  input: "main/l3/w"
  input: "main/l3/w/Initializer/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l3/w/read"
  op: "Identity"
  input: "main/l3/w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
}
node {
  name: "main/l3/Conv2D"
  op: "Conv2D"
  input: "main/Relu_1"
  input: "main/l3/w/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "main/l3/biases/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l3/biases"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l3/biases/Assign"
  op: "Assign"
  input: "main/l3/biases"
  input: "main/l3/biases/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l3/biases/read"
  op: "Identity"
  input: "main/l3/biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
}
node {
  name: "main/l3/BiasAdd"
  op: "BiasAdd"
  input: "main/l3/Conv2D"
  input: "main/l3/biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "main/Relu_2"
  op: "Relu"
  input: "main/l3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377@\014\000\000"
      }
    }
  }
}
node {
  name: "main/Reshape"
  op: "Reshape"
  input: "main/Relu_2"
  input: "main/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "main/l4/Matrix/Initializer/random_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "@\014\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "main/l4/Matrix/Initializer/random_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l4/Matrix/Initializer/random_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "main/l4/Matrix/Initializer/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "main/l4/Matrix/Initializer/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 60
    }
  }
}
node {
  name: "main/l4/Matrix/Initializer/random_normal/mul"
  op: "Mul"
  input: "main/l4/Matrix/Initializer/random_normal/RandomStandardNormal"
  input: "main/l4/Matrix/Initializer/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
}
node {
  name: "main/l4/Matrix/Initializer/random_normal"
  op: "Add"
  input: "main/l4/Matrix/Initializer/random_normal/mul"
  input: "main/l4/Matrix/Initializer/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
}
node {
  name: "main/l4/Matrix"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3136
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l4/Matrix/Assign"
  op: "Assign"
  input: "main/l4/Matrix"
  input: "main/l4/Matrix/Initializer/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l4/Matrix/read"
  op: "Identity"
  input: "main/l4/Matrix"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
}
node {
  name: "main/l4/bias/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 512
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/l4/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/l4/bias/Assign"
  op: "Assign"
  input: "main/l4/bias"
  input: "main/l4/bias/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/l4/bias/read"
  op: "Identity"
  input: "main/l4/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
}
node {
  name: "main/l4/MatMul"
  op: "MatMul"
  input: "main/Reshape"
  input: "main/l4/Matrix/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "main/l4/BiasAdd"
  op: "BiasAdd"
  input: "main/l4/MatMul"
  input: "main/l4/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "main/l4/Relu"
  op: "Relu"
  input: "main/l4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/q/Matrix/Initializer/random_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\002\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "main/q/Matrix/Initializer/random_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/q/Matrix/Initializer/random_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "main/q/Matrix/Initializer/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "main/q/Matrix/Initializer/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 76
    }
  }
}
node {
  name: "main/q/Matrix/Initializer/random_normal/mul"
  op: "Mul"
  input: "main/q/Matrix/Initializer/random_normal/RandomStandardNormal"
  input: "main/q/Matrix/Initializer/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
}
node {
  name: "main/q/Matrix/Initializer/random_normal"
  op: "Add"
  input: "main/q/Matrix/Initializer/random_normal/mul"
  input: "main/q/Matrix/Initializer/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
}
node {
  name: "main/q/Matrix"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/q/Matrix/Assign"
  op: "Assign"
  input: "main/q/Matrix"
  input: "main/q/Matrix/Initializer/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/q/Matrix/read"
  op: "Identity"
  input: "main/q/Matrix"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
}
node {
  name: "main/q/bias/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "main/q/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "main/q/bias/Assign"
  op: "Assign"
  input: "main/q/bias"
  input: "main/q/bias/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "main/q/bias/read"
  op: "Identity"
  input: "main/q/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
}
node {
  name: "main/q/MatMul"
  op: "MatMul"
  input: "main/l4/Relu"
  input: "main/q/Matrix/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "main/q/BiasAdd"
  op: "BiasAdd"
  input: "main/q/MatMul"
  input: "main/q/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "main/ArgMax/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "main/ArgMax"
  op: "ArgMax"
  input: "main/q/BiasAdd"
  input: "main/ArgMax/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "output_type"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "main/Mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "main/Mean"
  op: "Mean"
  input: "main/q/BiasAdd"
  input: "main/Mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "main/strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "main/strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "main/strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "main/strided_slice"
  op: "StridedSlice"
  input: "main/Mean"
  input: "main/strided_slice/stack"
  input: "main/strided_slice/stack_1"
  input: "main/strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "main/q/0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "main/q/0"
      }
    }
  }
}
node {
  name: "main/q/0"
  op: "HistogramSummary"
  input: "main/q/0/tag"
  input: "main/strided_slice"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/strided_slice_1/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "main/strided_slice_1/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "main/strided_slice_1/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "main/strided_slice_1"
  op: "StridedSlice"
  input: "main/Mean"
  input: "main/strided_slice_1/stack"
  input: "main/strided_slice_1/stack_1"
  input: "main/strided_slice_1/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "main/q/1/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "main/q/1"
      }
    }
  }
}
node {
  name: "main/q/1"
  op: "HistogramSummary"
  input: "main/q/1/tag"
  input: "main/strided_slice_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/strided_slice_2/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "main/strided_slice_2/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "main/strided_slice_2/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "main/strided_slice_2"
  op: "StridedSlice"
  input: "main/Mean"
  input: "main/strided_slice_2/stack"
  input: "main/strided_slice_2/stack_1"
  input: "main/strided_slice_2/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "main/q/2/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "main/q/2"
      }
    }
  }
}
node {
  name: "main/q/2"
  op: "HistogramSummary"
  input: "main/q/2/tag"
  input: "main/strided_slice_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/strided_slice_3/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "main/strided_slice_3/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "main/strided_slice_3/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "main/strided_slice_3"
  op: "StridedSlice"
  input: "main/Mean"
  input: "main/strided_slice_3/stack"
  input: "main/strided_slice_3/stack_1"
  input: "main/strided_slice_3/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "main/q/3/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "main/q/3"
      }
    }
  }
}
node {
  name: "main/q/3"
  op: "HistogramSummary"
  input: "main/q/3/tag"
  input: "main/strided_slice_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "main/Merge/MergeSummary"
  op: "MergeSummary"
  input: "main/q/0"
  input: "main/q/1"
  input: "main/q/2"
  input: "main/q/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
}
node {
  name: "target/target_s_t"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 84
        }
        dim {
          size: 84
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "target/target_l1/w/Initializer/truncated_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\010\000\000\000\010\000\000\000\004\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "target/target_l1/w/Initializer/truncated_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l1/w/Initializer/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "target/target_l1/w/Initializer/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "target/target_l1/w/Initializer/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 121
    }
  }
}
node {
  name: "target/target_l1/w/Initializer/truncated_normal/mul"
  op: "Mul"
  input: "target/target_l1/w/Initializer/truncated_normal/TruncatedNormal"
  input: "target/target_l1/w/Initializer/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
}
node {
  name: "target/target_l1/w/Initializer/truncated_normal"
  op: "Add"
  input: "target/target_l1/w/Initializer/truncated_normal/mul"
  input: "target/target_l1/w/Initializer/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
}
node {
  name: "target/target_l1/w"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 8
        }
        dim {
          size: 8
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l1/w/Assign"
  op: "Assign"
  input: "target/target_l1/w"
  input: "target/target_l1/w/Initializer/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l1/w/read"
  op: "Identity"
  input: "target/target_l1/w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
}
node {
  name: "target/target_l1/Conv2D"
  op: "Conv2D"
  input: "target/target_s_t"
  input: "target/target_l1/w/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 4
        i: 4
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l1/biases/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l1/biases"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l1/biases/Assign"
  op: "Assign"
  input: "target/target_l1/biases"
  input: "target/target_l1/biases/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l1/biases/read"
  op: "Identity"
  input: "target/target_l1/biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/biases"
      }
    }
  }
}
node {
  name: "target/target_l1/BiasAdd"
  op: "BiasAdd"
  input: "target/target_l1/Conv2D"
  input: "target/target_l1/biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "target/Relu"
  op: "Relu"
  input: "target/target_l1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "target/target_l2/w/Initializer/truncated_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "target/target_l2/w/Initializer/truncated_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l2/w/Initializer/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "target/target_l2/w/Initializer/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "target/target_l2/w/Initializer/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 137
    }
  }
}
node {
  name: "target/target_l2/w/Initializer/truncated_normal/mul"
  op: "Mul"
  input: "target/target_l2/w/Initializer/truncated_normal/TruncatedNormal"
  input: "target/target_l2/w/Initializer/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
}
node {
  name: "target/target_l2/w/Initializer/truncated_normal"
  op: "Add"
  input: "target/target_l2/w/Initializer/truncated_normal/mul"
  input: "target/target_l2/w/Initializer/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
}
node {
  name: "target/target_l2/w"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l2/w/Assign"
  op: "Assign"
  input: "target/target_l2/w"
  input: "target/target_l2/w/Initializer/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l2/w/read"
  op: "Identity"
  input: "target/target_l2/w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
}
node {
  name: "target/target_l2/Conv2D"
  op: "Conv2D"
  input: "target/Relu"
  input: "target/target_l2/w/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l2/biases/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l2/biases"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l2/biases/Assign"
  op: "Assign"
  input: "target/target_l2/biases"
  input: "target/target_l2/biases/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l2/biases/read"
  op: "Identity"
  input: "target/target_l2/biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/biases"
      }
    }
  }
}
node {
  name: "target/target_l2/BiasAdd"
  op: "BiasAdd"
  input: "target/target_l2/Conv2D"
  input: "target/target_l2/biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "target/Relu_1"
  op: "Relu"
  input: "target/target_l2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "target/target_l3/w/Initializer/truncated_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "target/target_l3/w/Initializer/truncated_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l3/w/Initializer/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "target/target_l3/w/Initializer/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "target/target_l3/w/Initializer/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 153
    }
  }
}
node {
  name: "target/target_l3/w/Initializer/truncated_normal/mul"
  op: "Mul"
  input: "target/target_l3/w/Initializer/truncated_normal/TruncatedNormal"
  input: "target/target_l3/w/Initializer/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
}
node {
  name: "target/target_l3/w/Initializer/truncated_normal"
  op: "Add"
  input: "target/target_l3/w/Initializer/truncated_normal/mul"
  input: "target/target_l3/w/Initializer/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
}
node {
  name: "target/target_l3/w"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l3/w/Assign"
  op: "Assign"
  input: "target/target_l3/w"
  input: "target/target_l3/w/Initializer/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l3/w/read"
  op: "Identity"
  input: "target/target_l3/w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
}
node {
  name: "target/target_l3/Conv2D"
  op: "Conv2D"
  input: "target/Relu_1"
  input: "target/target_l3/w/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l3/biases/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l3/biases"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l3/biases/Assign"
  op: "Assign"
  input: "target/target_l3/biases"
  input: "target/target_l3/biases/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l3/biases/read"
  op: "Identity"
  input: "target/target_l3/biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/biases"
      }
    }
  }
}
node {
  name: "target/target_l3/BiasAdd"
  op: "BiasAdd"
  input: "target/target_l3/Conv2D"
  input: "target/target_l3/biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "target/Relu_2"
  op: "Relu"
  input: "target/target_l3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "target/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377@\014\000\000"
      }
    }
  }
}
node {
  name: "target/Reshape"
  op: "Reshape"
  input: "target/Relu_2"
  input: "target/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "target/target_l4/Matrix/Initializer/random_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "@\014\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "target/target_l4/Matrix/Initializer/random_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l4/Matrix/Initializer/random_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "target/target_l4/Matrix/Initializer/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "target/target_l4/Matrix/Initializer/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 171
    }
  }
}
node {
  name: "target/target_l4/Matrix/Initializer/random_normal/mul"
  op: "Mul"
  input: "target/target_l4/Matrix/Initializer/random_normal/RandomStandardNormal"
  input: "target/target_l4/Matrix/Initializer/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
}
node {
  name: "target/target_l4/Matrix/Initializer/random_normal"
  op: "Add"
  input: "target/target_l4/Matrix/Initializer/random_normal/mul"
  input: "target/target_l4/Matrix/Initializer/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
}
node {
  name: "target/target_l4/Matrix"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3136
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l4/Matrix/Assign"
  op: "Assign"
  input: "target/target_l4/Matrix"
  input: "target/target_l4/Matrix/Initializer/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l4/Matrix/read"
  op: "Identity"
  input: "target/target_l4/Matrix"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
}
node {
  name: "target/target_l4/bias/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 512
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_l4/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_l4/bias/Assign"
  op: "Assign"
  input: "target/target_l4/bias"
  input: "target/target_l4/bias/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_l4/bias/read"
  op: "Identity"
  input: "target/target_l4/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/bias"
      }
    }
  }
}
node {
  name: "target/target_l4/MatMul"
  op: "MatMul"
  input: "target/Reshape"
  input: "target/target_l4/Matrix/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "target/target_l4/BiasAdd"
  op: "BiasAdd"
  input: "target/target_l4/MatMul"
  input: "target/target_l4/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "target/target_l4/Relu"
  op: "Relu"
  input: "target/target_l4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "target/target_q/Matrix/Initializer/random_normal/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\002\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "target/target_q/Matrix/Initializer/random_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_q/Matrix/Initializer/random_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.019999999552965164
      }
    }
  }
}
node {
  name: "target/target_q/Matrix/Initializer/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "target/target_q/Matrix/Initializer/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 123
    }
  }
  attr {
    key: "seed2"
    value {
      i: 187
    }
  }
}
node {
  name: "target/target_q/Matrix/Initializer/random_normal/mul"
  op: "Mul"
  input: "target/target_q/Matrix/Initializer/random_normal/RandomStandardNormal"
  input: "target/target_q/Matrix/Initializer/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
}
node {
  name: "target/target_q/Matrix/Initializer/random_normal"
  op: "Add"
  input: "target/target_q/Matrix/Initializer/random_normal/mul"
  input: "target/target_q/Matrix/Initializer/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
}
node {
  name: "target/target_q/Matrix"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_q/Matrix/Assign"
  op: "Assign"
  input: "target/target_q/Matrix"
  input: "target/target_q/Matrix/Initializer/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_q/Matrix/read"
  op: "Identity"
  input: "target/target_q/Matrix"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
}
node {
  name: "target/target_q/bias/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "target/target_q/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "target/target_q/bias/Assign"
  op: "Assign"
  input: "target/target_q/bias"
  input: "target/target_q/bias/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "target/target_q/bias/read"
  op: "Identity"
  input: "target/target_q/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/bias"
      }
    }
  }
}
node {
  name: "target/target_q/MatMul"
  op: "MatMul"
  input: "target/target_l4/Relu"
  input: "target/target_q/Matrix/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "target/target_q/BiasAdd"
  op: "BiasAdd"
  input: "target/target_q/MatMul"
  input: "target/target_q/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "target/outputs_idx"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: -1
        }
      }
    }
  }
}
node {
  name: "target/GatherNd"
  op: "GatherNd"
  input: "target/target_q/BiasAdd"
  input: "target/outputs_idx"
  attr {
    key: "Tindices"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tparams"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "pred_to_target/l2_b"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign"
  op: "Assign"
  input: "target/target_l2/biases"
  input: "pred_to_target/l2_b"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/q_w"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_1"
  op: "Assign"
  input: "target/target_q/Matrix"
  input: "pred_to_target/q_w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/l4_w"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3136
        }
        dim {
          size: 512
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_2"
  op: "Assign"
  input: "target/target_l4/Matrix"
  input: "pred_to_target/l4_w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/l1_b"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_3"
  op: "Assign"
  input: "target/target_l1/biases"
  input: "pred_to_target/l1_b"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/q_b"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_4"
  op: "Assign"
  input: "target/target_q/bias"
  input: "pred_to_target/q_b"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_q/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/l1_w"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 8
        }
        dim {
          size: 8
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_5"
  op: "Assign"
  input: "target/target_l1/w"
  input: "pred_to_target/l1_w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l1/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/l3_w"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_6"
  op: "Assign"
  input: "target/target_l3/w"
  input: "pred_to_target/l3_w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/l2_w"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_7"
  op: "Assign"
  input: "target/target_l2/w"
  input: "pred_to_target/l2_w"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l2/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/l3_b"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_8"
  op: "Assign"
  input: "target/target_l3/biases"
  input: "pred_to_target/l3_b"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l3/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "pred_to_target/l4_b"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
}
node {
  name: "pred_to_target/Assign_9"
  op: "Assign"
  input: "target/target_l4/bias"
  input: "pred_to_target/l4_b"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@target/target_l4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/target_q_t"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
      }
    }
  }
}
node {
  name: "optimizer/action"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
      }
    }
  }
}
node {
  name: "optimizer/action_onehot/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/action_onehot/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/action_onehot/depth"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "optimizer/action_onehot/on_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/action_onehot/off_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/action_onehot"
  op: "OneHot"
  input: "optimizer/action"
  input: "optimizer/action_onehot/depth"
  input: "optimizer/action_onehot/on_value"
  input: "optimizer/action_onehot/off_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "TI"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "axis"
    value {
      i: -1
    }
  }
}
node {
  name: "optimizer/mul"
  op: "Mul"
  input: "main/q/BiasAdd"
  input: "optimizer/action_onehot"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Q/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "optimizer/Q"
  op: "Sum"
  input: "optimizer/mul"
  input: "optimizer/Q/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/sub"
  op: "Sub"
  input: "optimizer/target_q_t"
  input: "optimizer/Q"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Abs"
  op: "Abs"
  input: "optimizer/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Less/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/Less"
  op: "Less"
  input: "optimizer/Abs"
  input: "optimizer/Less/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Square"
  op: "Square"
  input: "optimizer/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/mul_1/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "optimizer/mul_1"
  op: "Mul"
  input: "optimizer/mul_1/x"
  input: "optimizer/Square"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Abs_1"
  op: "Abs"
  input: "optimizer/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/sub_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "optimizer/sub_1"
  op: "Sub"
  input: "optimizer/Abs_1"
  input: "optimizer/sub_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Select"
  op: "Select"
  input: "optimizer/Less"
  input: "optimizer/mul_1"
  input: "optimizer/sub_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "optimizer/loss"
  op: "Mean"
  input: "optimizer/Select"
  input: "optimizer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/learning_rate_step"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0002500000118743628
      }
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/Cast"
  op: "Cast"
  input: "optimizer/learning_rate_step"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/Cast_1/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 500
      }
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/Cast_1"
  op: "Cast"
  input: "optimizer/ExponentialDecay/Cast_1/x"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/Cast_2/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9599999785423279
      }
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/truediv"
  op: "RealDiv"
  input: "optimizer/ExponentialDecay/Cast"
  input: "optimizer/ExponentialDecay/Cast_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/Floor"
  op: "Floor"
  input: "optimizer/ExponentialDecay/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/ExponentialDecay/Pow"
  op: "Pow"
  input: "optimizer/ExponentialDecay/Cast_2/x"
  input: "optimizer/ExponentialDecay/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/ExponentialDecay"
  op: "Mul"
  input: "optimizer/ExponentialDecay/learning_rate"
  input: "optimizer/ExponentialDecay/Pow"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/Maximum/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0002500000118743628
      }
    }
  }
}
node {
  name: "optimizer/Maximum"
  op: "Maximum"
  input: "optimizer/Maximum/x"
  input: "optimizer/ExponentialDecay"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "optimizer/gradients/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/gradients/Fill"
  op: "Fill"
  input: "optimizer/gradients/Shape"
  input: "optimizer/gradients/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Reshape"
  op: "Reshape"
  input: "optimizer/gradients/Fill"
  input: "optimizer/gradients/optimizer/loss_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Shape"
  op: "Shape"
  input: "optimizer/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Tile"
  op: "Tile"
  input: "optimizer/gradients/optimizer/loss_grad/Reshape"
  input: "optimizer/gradients/optimizer/loss_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Shape_1"
  op: "Shape"
  input: "optimizer/Select"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Shape_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Prod"
  op: "Prod"
  input: "optimizer/gradients/optimizer/loss_grad/Shape_1"
  input: "optimizer/gradients/optimizer/loss_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Prod_1"
  op: "Prod"
  input: "optimizer/gradients/optimizer/loss_grad/Shape_2"
  input: "optimizer/gradients/optimizer/loss_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Maximum"
  op: "Maximum"
  input: "optimizer/gradients/optimizer/loss_grad/Prod_1"
  input: "optimizer/gradients/optimizer/loss_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/floordiv"
  op: "FloorDiv"
  input: "optimizer/gradients/optimizer/loss_grad/Prod"
  input: "optimizer/gradients/optimizer/loss_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/Cast"
  op: "Cast"
  input: "optimizer/gradients/optimizer/loss_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/loss_grad/truediv"
  op: "RealDiv"
  input: "optimizer/gradients/optimizer/loss_grad/Tile"
  input: "optimizer/gradients/optimizer/loss_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Select_grad/zeros_like"
  op: "ZerosLike"
  input: "optimizer/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Select_grad/Select"
  op: "Select"
  input: "optimizer/Less"
  input: "optimizer/gradients/optimizer/loss_grad/truediv"
  input: "optimizer/gradients/optimizer/Select_grad/zeros_like"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Select_grad/Select_1"
  op: "Select"
  input: "optimizer/Less"
  input: "optimizer/gradients/optimizer/Select_grad/zeros_like"
  input: "optimizer/gradients/optimizer/loss_grad/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Select_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/optimizer/Select_grad/Select"
  input: "^optimizer/gradients/optimizer/Select_grad/Select_1"
}
node {
  name: "optimizer/gradients/optimizer/Select_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/optimizer/Select_grad/Select"
  input: "^optimizer/gradients/optimizer/Select_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/Select_grad/Select"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Select_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/optimizer/Select_grad/Select_1"
  input: "^optimizer/gradients/optimizer/Select_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/Select_grad/Select_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/Shape_1"
  op: "Shape"
  input: "optimizer/Square"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "optimizer/gradients/optimizer/mul_1_grad/Shape"
  input: "optimizer/gradients/optimizer/mul_1_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/mul"
  op: "Mul"
  input: "optimizer/gradients/optimizer/Select_grad/tuple/control_dependency"
  input: "optimizer/Square"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/Sum"
  op: "Sum"
  input: "optimizer/gradients/optimizer/mul_1_grad/mul"
  input: "optimizer/gradients/optimizer/mul_1_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/Reshape"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/mul_1_grad/Sum"
  input: "optimizer/gradients/optimizer/mul_1_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/mul_1"
  op: "Mul"
  input: "optimizer/mul_1/x"
  input: "optimizer/gradients/optimizer/Select_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/Sum_1"
  op: "Sum"
  input: "optimizer/gradients/optimizer/mul_1_grad/mul_1"
  input: "optimizer/gradients/optimizer/mul_1_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/Reshape_1"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/mul_1_grad/Sum_1"
  input: "optimizer/gradients/optimizer/mul_1_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/optimizer/mul_1_grad/Reshape"
  input: "^optimizer/gradients/optimizer/mul_1_grad/Reshape_1"
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/optimizer/mul_1_grad/Reshape"
  input: "^optimizer/gradients/optimizer/mul_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/mul_1_grad/Reshape"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_1_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/optimizer/mul_1_grad/Reshape_1"
  input: "^optimizer/gradients/optimizer/mul_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/mul_1_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/Shape"
  op: "Shape"
  input: "optimizer/Abs_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "optimizer/gradients/optimizer/sub_1_grad/Shape"
  input: "optimizer/gradients/optimizer/sub_1_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/Sum"
  op: "Sum"
  input: "optimizer/gradients/optimizer/Select_grad/tuple/control_dependency_1"
  input: "optimizer/gradients/optimizer/sub_1_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/Reshape"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/sub_1_grad/Sum"
  input: "optimizer/gradients/optimizer/sub_1_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/Sum_1"
  op: "Sum"
  input: "optimizer/gradients/optimizer/Select_grad/tuple/control_dependency_1"
  input: "optimizer/gradients/optimizer/sub_1_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/Neg"
  op: "Neg"
  input: "optimizer/gradients/optimizer/sub_1_grad/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/Reshape_1"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/sub_1_grad/Neg"
  input: "optimizer/gradients/optimizer/sub_1_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/optimizer/sub_1_grad/Reshape"
  input: "^optimizer/gradients/optimizer/sub_1_grad/Reshape_1"
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/optimizer/sub_1_grad/Reshape"
  input: "^optimizer/gradients/optimizer/sub_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/sub_1_grad/Reshape"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_1_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/optimizer/sub_1_grad/Reshape_1"
  input: "^optimizer/gradients/optimizer/sub_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/sub_1_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Square_grad/mul/x"
  op: "Const"
  input: "^optimizer/gradients/optimizer/mul_1_grad/tuple/control_dependency_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Square_grad/mul"
  op: "Mul"
  input: "optimizer/gradients/optimizer/Square_grad/mul/x"
  input: "optimizer/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Square_grad/mul_1"
  op: "Mul"
  input: "optimizer/gradients/optimizer/mul_1_grad/tuple/control_dependency_1"
  input: "optimizer/gradients/optimizer/Square_grad/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Abs_1_grad/Sign"
  op: "Sign"
  input: "optimizer/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Abs_1_grad/mul"
  op: "Mul"
  input: "optimizer/gradients/optimizer/sub_1_grad/tuple/control_dependency"
  input: "optimizer/gradients/optimizer/Abs_1_grad/Sign"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/AddN"
  op: "AddN"
  input: "optimizer/gradients/optimizer/Square_grad/mul_1"
  input: "optimizer/gradients/optimizer/Abs_1_grad/mul"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/Square_grad/mul_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/Shape"
  op: "Shape"
  input: "optimizer/target_q_t"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/Shape_1"
  op: "Shape"
  input: "optimizer/Q"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "optimizer/gradients/optimizer/sub_grad/Shape"
  input: "optimizer/gradients/optimizer/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/Sum"
  op: "Sum"
  input: "optimizer/gradients/AddN"
  input: "optimizer/gradients/optimizer/sub_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/Reshape"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/sub_grad/Sum"
  input: "optimizer/gradients/optimizer/sub_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/Sum_1"
  op: "Sum"
  input: "optimizer/gradients/AddN"
  input: "optimizer/gradients/optimizer/sub_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/Neg"
  op: "Neg"
  input: "optimizer/gradients/optimizer/sub_grad/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/Reshape_1"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/sub_grad/Neg"
  input: "optimizer/gradients/optimizer/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/optimizer/sub_grad/Reshape"
  input: "^optimizer/gradients/optimizer/sub_grad/Reshape_1"
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/optimizer/sub_grad/Reshape"
  input: "^optimizer/gradients/optimizer/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/sub_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/optimizer/sub_grad/Reshape_1"
  input: "^optimizer/gradients/optimizer/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/sub_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Shape"
  op: "Shape"
  input: "optimizer/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/add"
  op: "Add"
  input: "optimizer/Q/reduction_indices"
  input: "optimizer/gradients/optimizer/Q_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/mod"
  op: "FloorMod"
  input: "optimizer/gradients/optimizer/Q_grad/add"
  input: "optimizer/gradients/optimizer/Q_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/range"
  op: "Range"
  input: "optimizer/gradients/optimizer/Q_grad/range/start"
  input: "optimizer/gradients/optimizer/Q_grad/Size"
  input: "optimizer/gradients/optimizer/Q_grad/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Fill/value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Fill"
  op: "Fill"
  input: "optimizer/gradients/optimizer/Q_grad/Shape_1"
  input: "optimizer/gradients/optimizer/Q_grad/Fill/value"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/DynamicStitch"
  op: "DynamicStitch"
  input: "optimizer/gradients/optimizer/Q_grad/range"
  input: "optimizer/gradients/optimizer/Q_grad/mod"
  input: "optimizer/gradients/optimizer/Q_grad/Shape"
  input: "optimizer/gradients/optimizer/Q_grad/Fill"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Maximum"
  op: "Maximum"
  input: "optimizer/gradients/optimizer/Q_grad/DynamicStitch"
  input: "optimizer/gradients/optimizer/Q_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/floordiv"
  op: "FloorDiv"
  input: "optimizer/gradients/optimizer/Q_grad/Shape"
  input: "optimizer/gradients/optimizer/Q_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Reshape"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/sub_grad/tuple/control_dependency_1"
  input: "optimizer/gradients/optimizer/Q_grad/DynamicStitch"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/Q_grad/Tile"
  op: "Tile"
  input: "optimizer/gradients/optimizer/Q_grad/Reshape"
  input: "optimizer/gradients/optimizer/Q_grad/floordiv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/Shape"
  op: "Shape"
  input: "main/q/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/Shape_1"
  op: "Shape"
  input: "optimizer/action_onehot"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "optimizer/gradients/optimizer/mul_grad/Shape"
  input: "optimizer/gradients/optimizer/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/mul"
  op: "Mul"
  input: "optimizer/gradients/optimizer/Q_grad/Tile"
  input: "optimizer/action_onehot"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/Sum"
  op: "Sum"
  input: "optimizer/gradients/optimizer/mul_grad/mul"
  input: "optimizer/gradients/optimizer/mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/Reshape"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/mul_grad/Sum"
  input: "optimizer/gradients/optimizer/mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/mul_1"
  op: "Mul"
  input: "main/q/BiasAdd"
  input: "optimizer/gradients/optimizer/Q_grad/Tile"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/Sum_1"
  op: "Sum"
  input: "optimizer/gradients/optimizer/mul_grad/mul_1"
  input: "optimizer/gradients/optimizer/mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/Reshape_1"
  op: "Reshape"
  input: "optimizer/gradients/optimizer/mul_grad/Sum_1"
  input: "optimizer/gradients/optimizer/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/optimizer/mul_grad/Reshape"
  input: "^optimizer/gradients/optimizer/mul_grad/Reshape_1"
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/optimizer/mul_grad/Reshape"
  input: "^optimizer/gradients/optimizer/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "optimizer/gradients/optimizer/mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/optimizer/mul_grad/Reshape_1"
  input: "^optimizer/gradients/optimizer/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/q/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "optimizer/gradients/optimizer/mul_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "optimizer/gradients/main/q/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/optimizer/mul_grad/tuple/control_dependency"
  input: "^optimizer/gradients/main/q/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/optimizer/mul_grad/tuple/control_dependency"
  input: "^optimizer/gradients/main/q/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/optimizer/mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/q/BiasAdd_grad/BiasAddGrad"
  input: "^optimizer/gradients/main/q/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/q/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/q/MatMul_grad/MatMul"
  op: "MatMul"
  input: "optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency"
  input: "main/q/Matrix/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/q/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "main/l4/Relu"
  input: "optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/main/q/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/q/MatMul_grad/MatMul"
  input: "^optimizer/gradients/main/q/MatMul_grad/MatMul_1"
}
node {
  name: "optimizer/gradients/main/q/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/q/MatMul_grad/MatMul"
  input: "^optimizer/gradients/main/q/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/q/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/q/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/q/MatMul_grad/MatMul_1"
  input: "^optimizer/gradients/main/q/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/q/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "optimizer/gradients/main/q/MatMul_grad/tuple/control_dependency"
  input: "main/l4/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "optimizer/gradients/main/l4/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/l4/Relu_grad/ReluGrad"
  input: "^optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "optimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/l4/Relu_grad/ReluGrad"
  input: "^optimizer/gradients/main/l4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l4/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGrad"
  input: "^optimizer/gradients/main/l4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/MatMul_grad/MatMul"
  op: "MatMul"
  input: "optimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency"
  input: "main/l4/Matrix/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "main/Reshape"
  input: "optimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/l4/MatMul_grad/MatMul"
  input: "^optimizer/gradients/main/l4/MatMul_grad/MatMul_1"
}
node {
  name: "optimizer/gradients/main/l4/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/l4/MatMul_grad/MatMul"
  input: "^optimizer/gradients/main/l4/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l4/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l4/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l4/MatMul_grad/MatMul_1"
  input: "^optimizer/gradients/main/l4/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l4/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/Reshape_grad/Shape"
  op: "Shape"
  input: "main/Relu_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/main/Reshape_grad/Reshape"
  op: "Reshape"
  input: "optimizer/gradients/main/l4/MatMul_grad/tuple/control_dependency"
  input: "optimizer/gradients/main/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/main/Relu_2_grad/ReluGrad"
  op: "ReluGrad"
  input: "optimizer/gradients/main/Reshape_grad/Reshape"
  input: "main/Relu_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "optimizer/gradients/main/Relu_2_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/Relu_2_grad/ReluGrad"
  input: "^optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "optimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/Relu_2_grad/ReluGrad"
  input: "^optimizer/gradients/main/l3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/Relu_2_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGrad"
  input: "^optimizer/gradients/main/l3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/Conv2D_grad/Shape"
  op: "Shape"
  input: "main/Relu_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "optimizer/gradients/main/l3/Conv2D_grad/Shape"
  input: "main/l3/w/read"
  input: "optimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/Conv2D_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "main/Relu_1"
  input: "optimizer/gradients/main/l3/Conv2D_grad/Shape_1"
  input: "optimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInput"
  input: "^optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "optimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInput"
  input: "^optimizer/gradients/main/l3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilter"
  input: "^optimizer/gradients/main/l3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/Relu_1_grad/ReluGrad"
  op: "ReluGrad"
  input: "optimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependency"
  input: "main/Relu_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "optimizer/gradients/main/Relu_1_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/Relu_1_grad/ReluGrad"
  input: "^optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "optimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/Relu_1_grad/ReluGrad"
  input: "^optimizer/gradients/main/l2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/Relu_1_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGrad"
  input: "^optimizer/gradients/main/l2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/Conv2D_grad/Shape"
  op: "Shape"
  input: "main/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "optimizer/gradients/main/l2/Conv2D_grad/Shape"
  input: "main/l2/w/read"
  input: "optimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/Conv2D_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "main/Relu"
  input: "optimizer/gradients/main/l2/Conv2D_grad/Shape_1"
  input: "optimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInput"
  input: "^optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "optimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInput"
  input: "^optimizer/gradients/main/l2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilter"
  input: "^optimizer/gradients/main/l2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "optimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependency"
  input: "main/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "optimizer/gradients/main/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/Relu_grad/ReluGrad"
  input: "^optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "optimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/Relu_grad/ReluGrad"
  input: "^optimizer/gradients/main/l1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGrad"
  input: "^optimizer/gradients/main/l1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/Conv2D_grad/Shape"
  op: "Shape"
  input: "main/s_t"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "optimizer/gradients/main/l1/Conv2D_grad/Shape"
  input: "main/l1/w/read"
  input: "optimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 4
        i: 4
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/Conv2D_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\010\000\000\000\010\000\000\000\004\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "main/s_t"
  input: "optimizer/gradients/main/l1/Conv2D_grad/Shape_1"
  input: "optimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 4
        i: 4
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInput"
  input: "^optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "optimizer/gradients/main/l1/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInput"
  input: "^optimizer/gradients/main/l1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "optimizer/gradients/main/l1/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^optimizer/gradients/main/l1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 8
          }
          dim {
            size: 8
          }
          dim {
            size: 4
          }
          dim {
            size: 32
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 8
        }
        dim {
          size: 8
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l1/w/RMSProp"
  input: "optimizer/main/l1/w/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l1/w/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 8
          }
          dim {
            size: 8
          }
          dim {
            size: 4
          }
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 8
        }
        dim {
          size: 8
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l1/w/RMSProp_1"
  input: "optimizer/main/l1/w/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l1/w/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l1/w/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l1/biases/RMSProp"
  input: "optimizer/main/l1/biases/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l1/biases/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l1/biases/RMSProp_1"
  input: "optimizer/main/l1/biases/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l1/biases/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l1/biases/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
          dim {
            size: 4
          }
          dim {
            size: 32
          }
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l2/w/RMSProp"
  input: "optimizer/main/l2/w/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l2/w/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
          dim {
            size: 4
          }
          dim {
            size: 32
          }
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 4
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l2/w/RMSProp_1"
  input: "optimizer/main/l2/w/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l2/w/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l2/w/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l2/biases/RMSProp"
  input: "optimizer/main/l2/biases/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l2/biases/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l2/biases/RMSProp_1"
  input: "optimizer/main/l2/biases/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l2/biases/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l2/biases/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 64
          }
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l3/w/RMSProp"
  input: "optimizer/main/l3/w/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l3/w/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 64
          }
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l3/w/RMSProp_1"
  input: "optimizer/main/l3/w/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l3/w/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l3/w/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l3/biases/RMSProp"
  input: "optimizer/main/l3/biases/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l3/biases/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l3/biases/RMSProp_1"
  input: "optimizer/main/l3/biases/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l3/biases/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l3/biases/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3136
          }
          dim {
            size: 512
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3136
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l4/Matrix/RMSProp"
  input: "optimizer/main/l4/Matrix/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l4/Matrix/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3136
          }
          dim {
            size: 512
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3136
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l4/Matrix/RMSProp_1"
  input: "optimizer/main/l4/Matrix/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l4/Matrix/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l4/Matrix/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 512
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/l4/bias/RMSProp"
  input: "optimizer/main/l4/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/l4/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 512
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/l4/bias/RMSProp_1"
  input: "optimizer/main/l4/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/l4/bias/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/l4/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 512
          }
          dim {
            size: 4
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/q/Matrix/RMSProp"
  input: "optimizer/main/q/Matrix/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/q/Matrix/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 512
          }
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/q/Matrix/RMSProp_1"
  input: "optimizer/main/q/Matrix/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/q/Matrix/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/q/Matrix/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp/Assign"
  op: "Assign"
  input: "optimizer/main/q/bias/RMSProp"
  input: "optimizer/main/q/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp/read"
  op: "Identity"
  input: "optimizer/main/q/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "optimizer/main/q/bias/RMSProp_1"
  input: "optimizer/main/q/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "optimizer/main/q/bias/RMSProp_1/read"
  op: "Identity"
  input: "optimizer/main/q/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
}
node {
  name: "optimizer/RMSProp/decay"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "optimizer/RMSProp/momentum"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.949999988079071
      }
    }
  }
}
node {
  name: "optimizer/RMSProp/epsilon"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.009999999776482582
      }
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l1/w/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l1/w"
  input: "optimizer/main/l1/w/RMSProp"
  input: "optimizer/main/l1/w/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l1/biases/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l1/biases"
  input: "optimizer/main/l1/biases/RMSProp"
  input: "optimizer/main/l1/biases/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l1/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l2/w/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l2/w"
  input: "optimizer/main/l2/w/RMSProp"
  input: "optimizer/main/l2/w/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l2/biases/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l2/biases"
  input: "optimizer/main/l2/biases/RMSProp"
  input: "optimizer/main/l2/biases/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l2/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l3/w/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l3/w"
  input: "optimizer/main/l3/w/RMSProp"
  input: "optimizer/main/l3/w/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/w"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l3/biases/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l3/biases"
  input: "optimizer/main/l3/biases/RMSProp"
  input: "optimizer/main/l3/biases/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l3/biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l4/Matrix/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l4/Matrix"
  input: "optimizer/main/l4/Matrix/RMSProp"
  input: "optimizer/main/l4/Matrix/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l4/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/l4/bias/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/l4/bias"
  input: "optimizer/main/l4/bias/RMSProp"
  input: "optimizer/main/l4/bias/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/l4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/q/Matrix/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/q/Matrix"
  input: "optimizer/main/q/Matrix/RMSProp"
  input: "optimizer/main/q/Matrix/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/q/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/Matrix"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp/update_main/q/bias/ApplyRMSProp"
  op: "ApplyRMSProp"
  input: "main/q/bias"
  input: "optimizer/main/q/bias/RMSProp"
  input: "optimizer/main/q/bias/RMSProp_1"
  input: "optimizer/Maximum"
  input: "optimizer/RMSProp/decay"
  input: "optimizer/RMSProp/momentum"
  input: "optimizer/RMSProp/epsilon"
  input: "optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@main/q/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "optimizer/RMSProp"
  op: "NoOp"
  input: "^optimizer/RMSProp/update_main/l1/w/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/l1/biases/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/l2/w/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/l2/biases/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/l3/w/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/l3/biases/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/l4/Matrix/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/l4/bias/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/q/Matrix/ApplyRMSProp"
  input: "^optimizer/RMSProp/update_main/q/bias/ApplyRMSProp"
}
node {
  name: "summary/average.reward"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/average.reward_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/average.reward_1"
      }
    }
  }
}
node {
  name: "summary/average.reward_1"
  op: "ScalarSummary"
  input: "summary/average.reward_1/tags"
  input: "summary/average.reward"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/average.loss"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/average.loss_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/average.loss_1"
      }
    }
  }
}
node {
  name: "summary/average.loss_1"
  op: "ScalarSummary"
  input: "summary/average.loss_1/tags"
  input: "summary/average.loss"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/average.q"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/average.q_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/average.q_1"
      }
    }
  }
}
node {
  name: "summary/average.q_1"
  op: "ScalarSummary"
  input: "summary/average.q_1/tags"
  input: "summary/average.q"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/episode.max_reward"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/episode.max_reward_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/episode.max_reward_1"
      }
    }
  }
}
node {
  name: "summary/episode.max_reward_1"
  op: "ScalarSummary"
  input: "summary/episode.max_reward_1/tags"
  input: "summary/episode.max_reward"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/episode.min_reward"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/episode.min_reward_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/episode.min_reward_1"
      }
    }
  }
}
node {
  name: "summary/episode.min_reward_1"
  op: "ScalarSummary"
  input: "summary/episode.min_reward_1/tags"
  input: "summary/episode.min_reward"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/episode.avg_reward"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/episode.avg_reward_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/episode.avg_reward_1"
      }
    }
  }
}
node {
  name: "summary/episode.avg_reward_1"
  op: "ScalarSummary"
  input: "summary/episode.avg_reward_1/tags"
  input: "summary/episode.avg_reward"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/episode.num_of_game"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/episode.num_of_game_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/episode.num_of_game_1"
      }
    }
  }
}
node {
  name: "summary/episode.num_of_game_1"
  op: "ScalarSummary"
  input: "summary/episode.num_of_game_1/tags"
  input: "summary/episode.num_of_game"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/training.learning_rate"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/training.learning_rate_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/training.learning_rate_1"
      }
    }
  }
}
node {
  name: "summary/training.learning_rate_1"
  op: "ScalarSummary"
  input: "summary/training.learning_rate_1/tags"
  input: "summary/training.learning_rate"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/e"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/e_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/e_1"
      }
    }
  }
}
node {
  name: "summary/e_1"
  op: "ScalarSummary"
  input: "summary/e_1/tags"
  input: "summary/e"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/episode.rewards"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/episode.rewards_1/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/episode.rewards_1"
      }
    }
  }
}
node {
  name: "summary/episode.rewards_1"
  op: "HistogramSummary"
  input: "summary/episode.rewards_1/tag"
  input: "summary/episode.rewards"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "summary/episode.actions"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "summary/episode.actions_1/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "summary/episode.actions_1"
      }
    }
  }
}
node {
  name: "summary/episode.actions_1"
  op: "HistogramSummary"
  input: "summary/episode.actions_1/tag"
  input: "summary/episode.actions"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "init"
  op: "NoOp"
  input: "^step/step/Assign"
  input: "^main/l1/w/Assign"
  input: "^main/l1/biases/Assign"
  input: "^main/l2/w/Assign"
  input: "^main/l2/biases/Assign"
  input: "^main/l3/w/Assign"
  input: "^main/l3/biases/Assign"
  input: "^main/l4/Matrix/Assign"
  input: "^main/l4/bias/Assign"
  input: "^main/q/Matrix/Assign"
  input: "^main/q/bias/Assign"
  input: "^target/target_l1/w/Assign"
  input: "^target/target_l1/biases/Assign"
  input: "^target/target_l2/w/Assign"
  input: "^target/target_l2/biases/Assign"
  input: "^target/target_l3/w/Assign"
  input: "^target/target_l3/biases/Assign"
  input: "^target/target_l4/Matrix/Assign"
  input: "^target/target_l4/bias/Assign"
  input: "^target/target_q/Matrix/Assign"
  input: "^target/target_q/bias/Assign"
  input: "^optimizer/main/l1/w/RMSProp/Assign"
  input: "^optimizer/main/l1/w/RMSProp_1/Assign"
  input: "^optimizer/main/l1/biases/RMSProp/Assign"
  input: "^optimizer/main/l1/biases/RMSProp_1/Assign"
  input: "^optimizer/main/l2/w/RMSProp/Assign"
  input: "^optimizer/main/l2/w/RMSProp_1/Assign"
  input: "^optimizer/main/l2/biases/RMSProp/Assign"
  input: "^optimizer/main/l2/biases/RMSProp_1/Assign"
  input: "^optimizer/main/l3/w/RMSProp/Assign"
  input: "^optimizer/main/l3/w/RMSProp_1/Assign"
  input: "^optimizer/main/l3/biases/RMSProp/Assign"
  input: "^optimizer/main/l3/biases/RMSProp_1/Assign"
  input: "^optimizer/main/l4/Matrix/RMSProp/Assign"
  input: "^optimizer/main/l4/Matrix/RMSProp_1/Assign"
  input: "^optimizer/main/l4/bias/RMSProp/Assign"
  input: "^optimizer/main/l4/bias/RMSProp_1/Assign"
  input: "^optimizer/main/q/Matrix/RMSProp/Assign"
  input: "^optimizer/main/q/Matrix/RMSProp_1/Assign"
  input: "^optimizer/main/q/bias/RMSProp/Assign"
  input: "^optimizer/main/q/bias/RMSProp_1/Assign"
}
versions {
  producer: 24
}
